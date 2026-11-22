#!/usr/bin/env python3
"""QOBLIB portfolio solver: clamp asset/day blocks, solve each QUBO with SA.

Run from the Project directory (NumPy, SciPy, requests, dimod, dwave-samplers):
    python auto_runner_extensive.py solver_partitioned_sa.py --manual 0 --silent

Pass solver options after the runner's ``--`` separator, e.g. ``--reheats 2``.
No Divi, Maestro, or credentials are needed for the default CPU solver.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
import re
import time
from typing import Protocol

import dimod
import numpy as np
from dwave.samplers import SimulatedAnnealingSampler
from dwave.samplers.sa.sampler import default_beta_range

import qoblib_tools as qt
from asset_groups import BlockEvaluator, BlockResult, clamp_and_iterate, grouping_fn
from auto_worker import load_instance_dir
from big_blocks import block_form, solve_sa
from qaoa_signed_blocks import CODES, block_qubo, decode

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SEEDS = (1, 2, 3)


class BlockStage(Protocol):
    """Return a (reads, n_bits) binary array in label order 0..n_bits-1.

    ``starts`` contains the best feasible warm start, then any promising states
    from the previous stage. A future QAOA stage can ignore the starts or use
    them to initialize a circuit, and return measured bitstrings in this order.
    """

    def __call__(
        self, bqm: dimod.BinaryQuadraticModel, starts: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class AnnealStage:
    reads: int
    sweeps: int
    start_temperature_scale: float = 1.0
    restart_all_reads: bool = False

    def __call__(
        self, bqm: dimod.BinaryQuadraticModel, starts: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        labels = list(range(bqm.num_variables))
        # β is inverse temperature; a scale > 1 starts hotter.
        beta_hot, beta_cold = default_beta_range(bqm)
        samples = SimulatedAnnealingSampler().sample(
            bqm,
            num_reads=self.reads,
            num_sweeps=self.sweeps,
            beta_range=(beta_hot / self.start_temperature_scale, beta_cold),
            initial_states=(starts[: self.reads], labels),
            initial_states_generator="tile" if self.restart_all_reads else "random",
            seed=int(rng.integers(1, 2**31)),
        )
        return samples.record.sample[:, [samples.variables.index(v) for v in labels]]


def encode_holding(holding: np.ndarray, weights: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Encode a feasible incumbent in the signed QUBO's per-asset bit order."""
    values = (CODES @ weights.T).T + offsets[:, None]
    return CODES[(values == holding[:, None]).argmax(axis=1)].reshape(-1).astype(np.int8)


class PartitionSASolver:
    """Adapter from a sequence of BQM sampling stages to asset_groups.BlockSolver."""

    def __init__(
        self,
        stages: tuple[BlockStage, ...],
        seed: int,
        integer_polish: tuple[int, int] | None = None,
    ) -> None:
        self.stages = stages
        self.rng = np.random.default_rng(seed)
        self.integer_polish = integer_polish
        self.stage_calls = 0
        self.max_qubo_vars = 0

    def __call__(self, ev: BlockEvaluator, L, S, days, G: np.ndarray) -> BlockResult:
        # This solver uses one day per block, so the current net holding is a
        # feasible incumbent for every block. Multi-day blocks need a separate
        # feasible initializer if their current holdings differ between days.
        if len(days) != 1 or ev.model.ub != 3:
            raise ValueError("Signed SA blocks require one day and position bounds [-3, 3]")
        current = L[days[0], G] - S[days[0], G]
        best = current
        best_cost = ev.costs(L, S, days, G, current[None])[0][0]
        bqm, weights, offsets, _ = block_qubo(ev, L, S, days, G, mu=0.0)
        self.max_qubo_vars = max(self.max_qubo_vars, bqm.num_variables)
        labels = list(range(bqm.num_variables))
        best_bits = encode_holding(current, weights, offsets)
        starts = best_bits[None]

        for stage in self.stages:
            bits = np.asarray(stage(bqm, starts, self.rng))
            self.stage_calls += 1
            if bits.ndim != 2 or bits.shape[1] != bqm.num_variables or not np.all(
                (bits == 0) | (bits == 1)
            ):
                raise ValueError("Block stages must return a (reads, n_bits) binary array")
            if not len(bits):
                continue
            bits = bits.astype(np.int8, copy=False)
            holdings = decode(bits, weights, offsets)
            costs, feasible = ev.costs(L, S, days, G, holdings)
            if feasible.any():
                j = int(np.argmin(np.where(feasible, costs, np.inf)))
                if costs[j] < best_cost:
                    best, best_cost, best_bits = holdings[j], costs[j], bits[j]
            # Keep the feasible incumbent even when the lowest QUBO-energy
            # sample is infeasible. Let the next stage explore both basins.
            energies = bqm.energies((bits, labels))
            low_energy = bits[int(np.argmin(energies))]
            starts = best_bits[None] if np.array_equal(best_bits, low_energy) else np.stack(
                (best_bits, low_energy)
            )

        if self.integer_polish is not None:
            # Existing budget-preserving one-/two-asset moves can polish a
            # feasible binary-SA sample even when most other samples violate caps.
            chains, sweeps = self.integer_polish
            form = block_form(ev, L, S, days, G)
            best = solve_sa(form, best, self.rng, chains=chains, sweeps=sweeps)
        return BlockResult(best)


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return number


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    env = os.environ
    p.add_argument("--instance", default=env.get("AUTORUNNER_INSTANCE", "po_a010_t10_orig"))
    p.add_argument("--instance-dir", default=env.get("AUTORUNNER_INSTANCE_DIR"))
    p.add_argument("--assets", type=int, default=int(env["AUTORUNNER_ASSETS"]) if env.get("AUTORUNNER_ASSETS") else None)
    p.add_argument("--days", type=int, default=int(env["AUTORUNNER_DAYS"]) if env.get("AUTORUNNER_DAYS") else None)
    p.add_argument("--lambda", dest="lam", default=env.get("AUTORUNNER_LAMBDA", "1e-4"))
    p.add_argument("--budget", type=int, default=int(env.get("AUTORUNNER_BUDGET", "4")))
    p.add_argument("--cloud", type=parse_bool, nargs="?", const=True,
                   default=parse_bool(env.get("AUTORUNNER_CLOUD", "false")),
                   help="Accepted for compatibility; always runs locally")
    p.add_argument("--solver-seed", type=int,
                   default=int(env["AUTORUNNER_SOLVER_SEED"]) if env.get("AUTORUNNER_SOLVER_SEED") else None,
                   help="Single RNG seed override (default: run 1, 2, 3)")
    p.add_argument("--gpu", action=argparse.BooleanOptionalAction, default=True,
                   help="Accepted for compatibility; this solver uses CPU")
    p.add_argument("--gpu-device", type=int, help="Accepted for compatibility; this solver uses CPU")
    p.add_argument("--group-size", type=positive_int, default=6, help="Maximum assets per block (3 bits each)")
    p.add_argument("--group-method", choices=("neighbourhood", "random", "contiguous"), default="neighbourhood")
    p.add_argument("--block-sweeps", type=positive_int, default=4, help="Maximum passes through all blocks per seed")
    p.add_argument("--patience", type=positive_int, default=2, help="Stop after this many non-improving passes")
    p.add_argument("--sa-reads", type=positive_int, default=24)
    p.add_argument("--sa-sweeps", type=positive_int, default=120, help="SA sweeps per read per stage")
    p.add_argument("--reheats", type=int, default=1, help="Extra warm-started stages per block; 0 disables reheats")
    p.add_argument("--reheat-temperature-scale", type=float, default=2.0, help="Initial temperature multiplier for reheats")
    p.add_argument("--integer-polish", action="store_true", help="Finish each block with constrained integer SA")
    return p


def resolve_solution_path(instance: str, base: str, budget: int, lam: str) -> Path:
    if re.search(r"_b\d+_l", instance):
        return Path(f"{instance}.sol")
    lam_tag = lam.replace("-0", "-").replace("+0", "+")
    return Path(f"{base}_b{budget:03d}_l{lam_tag}.sol")


def load_instance(instance: str, instance_dir: Path | None) -> qt.PortfolioInstance:
    base = re.sub(r"_b\d+_l.*$", "", instance.strip())
    if instance_dir is not None and instance_dir.resolve().name.startswith("po_"):
        base = instance_dir.resolve().name
    for directory in (instance_dir, SCRIPT_DIR / "instances" / base):
        if directory is not None and all(
            (directory / name).is_file() for name in ("stock_prices.txt.gz", "covariance_matrices.txt.gz")
        ):
            return load_instance_dir(directory, base)
    return qt.load_portfolio_instance(base)


def main(argv: list[str] | None = None) -> int:
    p = build_arg_parser()
    args = p.parse_args(argv)
    if args.reheats < 0 or not (1 <= args.reheat_temperature_scale < float("inf")):
        p.error("--reheats must be nonnegative and --reheat-temperature-scale must be finite and at least 1")
    if args.solver_seed is not None and args.solver_seed < 0:
        p.error("--solver-seed must be nonnegative")

    started = time.perf_counter()
    inst = load_instance(args.instance, Path(args.instance_dir) if args.instance_dir else None)
    if args.assets is not None and inst.n_assets != args.assets:
        raise ValueError(f"Asset count mismatch for {inst.name}: expected {args.assets}, got {inst.n_assets}")
    if args.days is not None and inst.n_periods != args.days:
        raise ValueError(f"Day/period count mismatch for {inst.name}: expected {args.days}, got {inst.n_periods}")
    if args.cloud:
        print("--cloud was requested; partitioned SA still runs locally on CPU.", flush=True)

    model = qt.PortfolioModel(inst, budget=args.budget, lam=args.lam)
    if not model.score({}).feasible:
        raise ValueError("The empty portfolio is infeasible for this budget; cannot initialize the block solver")
    stages = (AnnealStage(args.sa_reads, args.sa_sweeps),) + (
        AnnealStage(args.sa_reads, args.sa_sweeps, args.reheat_temperature_scale, restart_all_reads=True),
    ) * args.reheats
    groups = grouping_fn(model, args.group_size, args.group_method)
    seeds = DEFAULT_SEEDS if args.solver_seed is None else (args.solver_seed,)
    seed_runs = []
    best = None
    max_vars = 0
    for seed in seeds:
        t0 = time.perf_counter()
        solver = PartitionSASolver(
            stages, seed, (args.sa_reads, args.sa_sweeps) if args.integer_polish else None,
        )
        # Vary both the random sampler and the partition sequence by seed.
        result = clamp_and_iterate(
            model, groups, solver=solver, max_sweeps=args.block_sweeps,
            patience=args.patience, sweep_offset=seed, verbose=False,
        )
        max_vars = max(max_vars, solver.max_qubo_vars)
        score = int(result["score"])
        seed_runs.append({
            "seed": seed, "objective": score, "feasible": True,
            "sub_programs": result["blocks"], "circuits": 0,
            "duration_seconds": round(time.perf_counter() - t0, 3),
            "stage_calls": solver.stage_calls, "accepted_blocks": result["accepted"],
            "block_sweeps": len(result["history"]) - 1,
        })
        print(f"seed {seed}: {score} ({result['blocks']} block solves, {solver.stage_calls} sampling stages)", flush=True)
        if best is None or score < best[0]:
            best = (score, seed, result["L"], result["S"])

    assert best is not None
    best_obj, best_seed, L, S = best
    positions = model.to_positions(L, S)
    score = model.score(positions)
    if not score.feasible or score.objective != best_obj:
        raise RuntimeError("Final portfolio did not pass official feasibility and objective checks")
    sol_path = resolve_solution_path(args.instance, inst.name, args.budget, args.lam)
    written = model.write_solution(sol_path, positions, comment="QOBLIB: partitioned simulated annealing")
    payload = {
        "score": int(written.objective),
        "extra": {
            "instance": args.instance, "base_instance": inst.name,
            "assets": inst.n_assets, "days": inst.n_periods, "budget": args.budget, "lambda": args.lam,
            "cloud": False, "cloud_requested": args.cloud, "gpu_used": False,
            "gpu_backend": None, "gpu_device": None, "device": "cpu",
            "simulator_backend": "dwave_simulated_annealing",
            "backend_detail": "Local CPU; no quantum circuits executed",
            "feasible": bool(written.feasible), "breakdown": written.breakdown,
            "violations": list(written.violations), "solution_file": sol_path.name,
            "routine": "partitioned_sa", "cluster_size": 3 * args.group_size,
            "n_qubo_vars": None, "max_block_qubo_vars": max_vars,  # no global QUBO built
            "best_seed": best_seed, "total_circuits": 0, "seed_runs": seed_runs,
            "total_duration_seconds": round(time.perf_counter() - started, 3),
            "sa": {
                "reads": args.sa_reads, "sweeps": args.sa_sweeps, "reheats": args.reheats,
                "reheat_temperature_scale": args.reheat_temperature_scale,
                "group_method": args.group_method, "group_size_assets": args.group_size,
                "max_block_sweeps": args.block_sweeps, "patience": args.patience,
                "integer_polish": args.integer_polish,
            },
        },
    }
    print(f"Wrote {sol_path}: {written}")
    print(f"AUTORUNNER_RESULT: {json.dumps(payload)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
