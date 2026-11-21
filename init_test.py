#!/usr/bin/env python
"""Initialisation test for QOBLIB 06 (portfolio): one (instance, λ) per call.

Compares initialisations, everything else fixed, 3 seeds each, and keeps every solution:

    QAOA, partitioned (notebook 01 §1.7):  qaoa_random, qaoa_tqa, qaoa_linramp,
                                           iter_interp, iter_fourier, iter_chebyshev
    PCE, whole problem:                    pce_random, pce_progressive  (Divi's own PCE inits)

PCE uses MultiBasisEncoding(order=3), the cubic Pauli-correlation encoding of Sciorilli et al.
(arXiv:2401.09421): 13 qubits for 710 variables, 20 for 3110 (PolyEncoding would need 38 / 79).

Called by ``auto_runner.py`` (see AutoRunner.md) from the run directory:

    python init_test.py --instance po_a010_t10_orig_b004_l1e-4 --data-dir ../../instances/po_a010_t10_orig \
        --assets 10 --days 10 --lam 1e-4 --budget 4 --cloud false [--quick] [--variants ...]

Writes, relative to the working directory:

    po_a010_t10_orig_b004/config.txt                     fixed settings (once per problem)
    po_a010_t10_orig_b004/l1e-4/<stem>__<variant>__seed<s>.sol
    po_a010_t10_orig_b004/l1e-4/chosen.txt               initial values each variant picked
    po_a010_t10_orig_b004_l1e-4.sol                      best of all runs (official file name)
    po_a010_t10_orig_b004_l1e-4.result.json              {"score", "best_sol", "extra"}

``python init_test.py --postprocess RUN_DIR RUN_JSON`` draws one objective-vs-λ plot per problem and
writes, next to it, ``<problem>_objective_vs_lambda.csv`` (the plotted points) and
``<problem>_all_runs.csv`` (every run, every seed).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import warnings
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
warnings.filterwarnings("ignore", category=SyntaxWarning)

import qoblib_tools as qt  # noqa: E402  (the referee; never edited)

DIVI_VERSION = "0.16.2"  # the init injection uses a private Divi hook: pin the version

QAOA_VARIANTS = ["qaoa_random", "qaoa_tqa", "qaoa_linramp", "iter_interp", "iter_fourier", "iter_chebyshev"]
PCE_VARIANTS = ["pce_random", "pce_progressive"]
VARIANTS = QAOA_VARIANTS + PCE_VARIANTS
DEFAULT_VARIANTS = VARIANTS

# Everything except the initialisation is fixed (notebook 01 §1.5 and §1.7, with p = 4).
CONFIG = {
    "penalty_weight_W": 100.0,      # notebook 01 §1.7 (W_BIG)
    "normalize_bqm": True,
    "shots": 10_000,
    "population": 50,               # MonteCarloOptimizer population_size = rows of initial_params
    "iterations": 10,               # Monte Carlo iterations (QAOA, PCE)
    "p": 4,                         # QAOA layers
    "cluster_size": 12,             # max variables per partition (CommunityDecomposer)
    "beam_width": 10,
    "n_partition_candidates": 5,
    "top_polish": 10,               # candidates decoded + repaired + local-searched
    "iter_max_depth": 4,
    "iter_schedule": {1: 6, 2: 5, 3: 4, 4: 3},  # Σ depth·iters = 40 = 4 × 10 (same cost as p=4, 10 iters)
    "pce_encoding": "MultiBasisEncoding(order=3)",
    "pce_multibasis_order": 3,
    "pce_layers": 3,                # notebook 01 §1.5
    "pce_objective": "encoding default: SoftObjective(alpha=1.5*q**(k//2))",
    "pce_top": 20,                  # PCE samples ranked by energy (MultiBasis returns 1)
    "progressive_max_steps": 4,
    "progressive_iters_per_step": 3,  # initial_alpha 3.0, threshold 0.9, max_restarts 0: Divi defaults
    "tqa_dt_range": [0.05, 8.0],    # geomspace over the population, β negative
    "linramp_dgamma_range": [0.1, 8.0],   # geomspace, 7 values
    "linramp_dbeta_range": [-1.5, -0.1],  # linspace, 7 values
    "seeds": [1, 2, 3],
}
QUICK = {  # smoke tests only
    "population": 4, "iterations": 1, "p": 1, "iter_max_depth": 2, "iter_schedule": {1: 1, 2: 1},
    "progressive_max_steps": 1, "progressive_iters_per_step": 1, "seeds": [1], "shots": 1000,
    "top_polish": 3, "pce_top": 5,
}


# ── instance and QUBO ──────────────────────────────────────────────────────────────────


def load_instance_dir(path: Path, name: str) -> qt.PortfolioInstance:
    """Parse the instance files in ``path`` exactly like ``qt.load_portfolio_instance``."""
    symbols, index, entries = [], {}, []
    for day, sym, price in qt._data_rows(path / "stock_prices.txt.gz"):
        if sym not in index:
            index[sym] = len(symbols)
            symbols.append(sym)
        entries.append((index[sym], int(day), Fraction(price)))
    n_periods = max(t for _, t, _ in entries) + 1
    prices = [[None] * n_periods for _ in symbols]
    for i, t, p in entries:
        prices[i][t] = p
    if any(p is None for row in prices for p in row):
        raise ValueError(f"{name}: missing prices")
    cov = {}
    for day, s1, s2, value in qt._data_rows(path / "covariance_matrices.txt.gz"):
        t = int(day)
        if s1 in index and s2 in index and t < n_periods:
            cov[index[s1], index[s2], t] = Fraction(value)
    return qt.PortfolioInstance(name, symbols, n_periods, prices, cov)


def penalised(q: qt.PortfolioQUBO, weight: float, normalize: bool = False):
    """objective + weight·penalty as one BQM (notebook 01 §1.2)."""
    bqm = q.objective.copy()
    penalty = q.penalty.copy()
    penalty.scale(weight)
    bqm.update(penalty)
    if normalize:
        bqm.normalize()
    return bqm


def best_polished(model: qt.PortfolioModel, q: qt.PortfolioQUBO, samples, top: int):
    """Decode the first ``top`` samples, repair + local-search each; best (positions, score)."""
    candidates = [model.local_search(q.decode(s)) for s in list(samples)[:top]]
    best = min(candidates, key=lambda p: model.score(p).objective)
    return best, model.score(best)


def raw_best(model, q, samples) -> tuple[int | None, int]:
    """Best official objective among the *feasible* raw samples (before repair), and their count."""
    scores = [model.score(q.decode(s)) for s in samples]
    feasible = [s.objective for s in scores if s.feasible]
    return (min(feasible) if feasible else None), len(feasible)


def make_backend(cloud: bool, shots: int, seed: int):
    from divi.backends import JobConfig, MaestroSimulator, QoroService

    if cloud:
        return QoroService(job_config=JobConfig(shots=shots, simulator_cluster="qoro_maestro"))
    backend = MaestroSimulator(shots=shots)
    # Without this the shot sampling is unseeded and the same program seed gives different results.
    # Divi derives a per-circuit seed from (seed, label, QASM), so thread order does not matter.
    backend.set_seed(seed)
    return backend


# ── initial parameters ─────────────────────────────────────────────────────────────────


def _ramp(p: int, d_gamma: float, d_beta: float) -> np.ndarray:
    """Linear ramp, interleaved per layer as Divi expects: [γ0, β0, γ1, β1, …]."""
    s = (np.arange(p) + 0.5) / p
    row = np.empty(2 * p)
    row[0::2] = s * d_gamma
    row[1::2] = (1 - s) * d_beta
    return row


def qaoa_init_rows(variant: str, p: int, population: int, cfg: dict):
    """(rows, labels) for TQA / linear ramp, or (None, None) for Divi's own initialisation."""
    if variant == "qaoa_tqa":
        # β negative: Divi's mixer is +ΣX on |+⟩ (confirmed: all 70 sub-programs chose β < 0).
        dts = np.geomspace(*cfg["tqa_dt_range"], population)
        return np.array([_ramp(p, d, -d) for d in dts]), [{"dt": float(d)} for d in dts]
    if variant == "qaoa_linramp":
        k = max(1, int(math.isqrt(population)))
        dgs = np.geomspace(*cfg["linramp_dgamma_range"], k)
        dbs = np.linspace(*cfg["linramp_dbeta_range"], k)
        rows, labels = [], []
        for dg in dgs:
            for db in dbs:
                rows.append(_ramp(p, dg, db))
                labels.append({"d_gamma": float(dg), "d_beta": float(db)})
        extra = np.geomspace(0.5, 4.0, max(population - len(rows), 1))
        for d in extra[: population - len(rows)]:  # fill to the population with TQA-like rows
            rows.append(_ramp(p, d, -d))
            labels.append({"d_gamma": float(d), "d_beta": float(-d)})
        return np.array(rows[:population]), labels[:population]
    return None, None


def chosen_rows(program, rows: np.ndarray | None, labels) -> list:
    """Label of the initial row with the lowest loss in the first Monte Carlo iteration."""
    if rows is None:
        return []
    try:
        first = program.losses_history[0]
        evaluated = program.param_history("all_evaluated")[0]
    except Exception:
        return []
    losses = np.array([first[str(i)] for i in range(len(first))], dtype=float)
    x = np.atleast_2d(evaluated)[int(np.nanargmin(losses))]
    return [labels[int(np.argmin(np.linalg.norm(rows - x, axis=1)))]]


# ── one run ────────────────────────────────────────────────────────────────────────────


def run_qaoa(variant: str, bqm, seed: int, cfg: dict, backend):
    import hybrid
    from divi.qprog.algorithms import InterpolationStrategy
    from divi.qprog.ensembles import BeamSearchStrategy, PartitioningProgramEnsemble
    from divi.qprog.optimizers import MonteCarloOptimizer
    from divi.qprog.problems import BinaryOptimizationProblem, CommunityDecomposer, QUBOPartitioningConfig

    problem = BinaryOptimizationProblem(
        bqm,
        decomposer=CommunityDecomposer(QUBOPartitioningConfig(max_n_variables_per_cluster=cfg["cluster_size"])),
        composer=hybrid.SplatComposer(),
        local_search=True,
    )
    kwargs = {}
    if variant.startswith("iter_"):
        routine, n_layers = "iterative_qaoa", cfg["iter_max_depth"]
        schedule = dict(cfg["iter_schedule"])
        kwargs = dict(strategy=InterpolationStrategy[variant.split("_", 1)[1].upper()],
                      max_iterations_per_depth=lambda d, s=schedule: s[d],
                      convergence_threshold=None)
    else:
        routine, n_layers = "qaoa", cfg["p"]
    ensemble = PartitioningProgramEnsemble(
        problem=problem, quantum_routine=routine, n_layers=n_layers,
        optimizer=MonteCarloOptimizer(population_size=cfg["population"]),
        max_iterations=cfg["iterations"], backend=backend, seed=seed, **kwargs,
    )
    rows, labels = qaoa_init_rows(variant, cfg["p"], cfg["population"], cfg)
    ensemble.create_programs()
    if rows is not None:
        # Divi's ensemble starts each sub-program with run() and no initial_params; every VQA then
        # asks _initialize_param_sets() for its starting population, so override it per program.
        for prog in ensemble.programs.values():
            expected = tuple(prog.get_expected_param_shape())
            if rows.shape != expected:
                raise ValueError(f"init rows {rows.shape} != expected {expected}")
            prog._initialize_param_sets = (lambda r=rows: r.copy())
    ensemble.run()
    tops = ensemble.get_top_solutions(
        n=cfg["top_polish"],
        strategy=BeamSearchStrategy(beam_width=cfg["beam_width"], n_partition_candidates=cfg["n_partition_candidates"]),
    )
    samples = [dict(zip(bqm.variables, solution)) for solution, _ in tops]
    chosen = [c for prog in ensemble.programs.values() for c in chosen_rows(prog, rows, labels)]
    return samples, chosen, {"circuits": ensemble.total_circuit_count,
                             "backend_time_s": ensemble.total_run_time,
                             "n_subprograms": len(ensemble.programs)}


def run_pce(variant: str, bqm, seed: int, cfg: dict, backend):
    """PCE on the whole QUBO, with Divi's own initialisations only (no initial_params):
    ``pce_random`` = uniform [0, 2π) (VariationalQuantumAlgorithm default),
    ``pce_progressive`` = ProgressivePCE, each α step warm-started from the previous best."""
    from divi.qprog.algorithms import PCE, MultiBasisEncoding, ProgressivePCE
    from divi.qprog.optimizers import MonteCarloOptimizer
    from divi.qprog.problems import BinaryOptimizationProblem

    problem = BinaryOptimizationProblem(bqm)
    encoding = MultiBasisEncoding(order=cfg["pce_multibasis_order"])
    common = dict(encoding=encoding, n_layers=cfg["pce_layers"],
                  optimizer=MonteCarloOptimizer(population_size=cfg["population"]), backend=backend, seed=seed)
    if variant == "pce_progressive":
        prog = ProgressivePCE(problem, max_steps=cfg["progressive_max_steps"],
                              max_iterations_per_step=cfg["progressive_iters_per_step"], **common)
    else:  # objective=None → the encoding's default SoftObjective(alpha = 1.5·q^(k//2))
        prog = PCE(problem, max_iterations=cfg["iterations"], **common)
    prog.run()
    tops = prog.get_top_solutions(n=cfg["pce_top"], min_prob=0.0, sort_by="energy")
    samples = [s.bitstring for s in tops]
    return samples, [], {"circuits": prog.total_circuit_count, "backend_time_s": prog.total_run_time,
                         "n_qubits": prog.n_qubits}


# ── output helpers ─────────────────────────────────────────────────────────────────────


def _atomic_write(path: Path, text: str, only_if_missing: bool = False) -> None:
    if only_if_missing and path.exists():
        return
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    if only_if_missing and path.exists():
        tmp.unlink()
        return
    os.replace(tmp, path)


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True,
                              text=True, timeout=10).stdout.strip() or None
    except Exception:
        return None


def _summarise_chosen(chosen: list[dict]) -> str:
    if not chosen:
        return "n/a (Divi's own initialisation)"
    keys = list(chosen[0])
    numeric = list(keys)
    parts = [f"{k}: median {np.median([c[k] for c in chosen]):.4g}" for k in numeric]
    counts = Counter(tuple((k, round(c[k], 4)) for k in keys) for c in chosen)
    hist = ", ".join(f"{dict(k)}×{n}" for k, n in counts.most_common(8))
    return f"{len(chosen)} program(s); " + "; ".join(parts) + f"\n      most common: {hist}"


# ── main ───────────────────────────────────────────────────────────────────────────────


def run_worker(args) -> int:
    from importlib.metadata import version

    divi = version("qoro-divi")
    if divi != DIVI_VERSION:
        raise SystemExit(f"init_test.py needs qoro-divi {DIVI_VERSION} (found {divi}): it overrides a private hook")

    cfg = dict(CONFIG)
    if args.quick:
        cfg.update(QUICK)
    if args.seeds:
        cfg["seeds"] = args.seeds
    variants = args.variants or DEFAULT_VARIANTS
    cloud = args.cloud.lower() in ("1", "true", "yes")

    m = re.match(r"^(po_a\d+_t\d+_(?:orig|s\d\d))_b(\d+)_l(.+)$", args.instance)
    if not m:
        raise SystemExit(f"cannot parse --instance {args.instance!r}")
    po_base = m.group(1)
    inst = load_instance_dir(Path(args.data_dir), po_base)
    if len(inst.symbols) != args.assets or inst.n_periods != args.days:
        print(f"WARNING: data has {len(inst.symbols)} assets × {inst.n_periods} days, "
              f"expected {args.assets} × {args.days}", flush=True)
    model = qt.PortfolioModel(inst, budget=args.budget, lam=args.lam)
    q = qt.PortfolioQUBO(model)
    bqm = penalised(q, cfg["penalty_weight_W"], normalize=cfg["normalize_bqm"])

    problem = f"{po_base}_b{args.budget:03d}"
    stem = f"{problem}_l{args.lam}"
    lam_dir = Path(problem) / f"l{args.lam}"
    lam_dir.mkdir(parents=True, exist_ok=True)
    config_txt = {
        "problem": problem, "solvers": {"qaoa_*/iter_*": "partitioned QAOA (PartitioningProgramEnsemble)",
                                        "pce_*": "PCE on the whole QUBO"},
        "variants": variants, "quick": args.quick, "cloud": cloud, "qoro-divi": divi,
        "git_commit": _git_commit(), "n_vars": q.n_vars, **cfg,
    }
    _atomic_write(Path(problem) / "config.txt", json.dumps(config_txt, indent=2, default=str) + "\n",
                  only_if_missing=True)

    credits = None
    if cloud:
        try:
            from divi.backends import QoroService

            credits = {"before": QoroService().get_credit_balance()}
        except Exception as err:
            credits = {"error": str(err)}

    print(f"{args.instance}: {q.n_vars} QUBO variables, variants {variants}, seeds {cfg['seeds']}, "
          f"{'cloud' if cloud else 'local'}{' (QUICK)' if args.quick else ''}", flush=True)
    runs = []
    result_path = Path(f"{args.instance}.result.json")

    def dump(best=None):
        extra = {"config": config_txt, "runs": runs, "credits": credits}
        payload = {"score": best["objective"] if best else None,
                   "best_sol": f"{args.instance}.sol" if best else None, "extra": extra}
        _atomic_write(result_path, json.dumps(payload, indent=2, default=str))
        return payload

    for variant in variants:
        solver = "qaoa_partitioned" if variant in QAOA_VARIANTS else "pce_whole"
        for seed in cfg["seeds"]:
            rec = {"variant": variant, "solver": solver, "seed": seed, "objective": None, "feasible": None,
                   "raw_best_feasible": None, "raw_n_feasible": None, "sol": None, "chosen": [],
                   "wall_s": None, "error": None}
            t0 = time.time()
            try:
                backend = make_backend(cloud, cfg["shots"], seed)
                runner = run_qaoa if variant in QAOA_VARIANTS else run_pce
                samples, chosen, stats = runner(variant, bqm, seed, cfg, backend)
                rec.update(stats)
                rec["chosen"] = chosen
                rec["raw_best_feasible"], rec["raw_n_feasible"] = raw_best(model, q, samples)
                positions, score = best_polished(model, q, samples, cfg["top_polish"])
                sol = lam_dir / f"{stem}__{variant}__seed{seed}.sol"
                model.write_solution(sol, positions,
                                     comment=f"init_test variant={variant} seed={seed}{' QUICK' if args.quick else ''}")
                rec.update(objective=int(score.objective), feasible=bool(score.feasible), sol=sol.as_posix(),
                           breakdown=dict(score.breakdown))
            except Exception as err:
                rec["error"] = f"{type(err).__name__}: {err}"
                rec["traceback"] = traceback.format_exc()[-4000:]
            rec["wall_s"] = round(time.time() - t0, 2)
            runs.append(rec)
            status = rec["error"] or f"objective {rec['objective']} ({'feasible' if rec['feasible'] else 'INFEASIBLE'})"
            print(f"  {variant:16s} seed {seed}: {status}  [{rec['wall_s']}s, {rec.get('circuits')} circuits]",
                  flush=True)
            dump()

    # chosen.txt: which initial values the first Monte Carlo iteration preferred.
    lines = [f"# {stem}: initial row with the lowest loss in the first Monte Carlo iteration", ""]
    for variant in variants:
        for rec in (r for r in runs if r["variant"] == variant):
            lines.append(f"{variant:16s} seed {rec['seed']}: {_summarise_chosen(rec['chosen'])}")
    _atomic_write(lam_dir / "chosen.txt", "\n".join(lines) + "\n")

    if cloud and credits is not None and "before" in credits:
        try:
            from divi.backends import QoroService

            credits["after"] = QoroService().get_credit_balance()
        except Exception as err:
            credits["after_error"] = str(err)

    ok = [r for r in runs if r["objective"] is not None]
    feasible = [r for r in ok if r["feasible"]] or ok
    best = min(feasible, key=lambda r: r["objective"]) if feasible else None
    if best:
        shutil.copyfile(best["sol"], f"{args.instance}.sol")
    payload = dump(best)
    print(f"best: {best['variant']} seed {best['seed']} → {best['objective']}" if best else "no solution", flush=True)
    print("AUTORUNNER_RESULT " + json.dumps({"score": payload["score"], "best_sol": payload["best_sol"]}), flush=True)
    return 0 if best else 1


# ── post-processing: one plot per problem ──────────────────────────────────────────────

# tab10, one fixed colour per variant (same variant = same colour in every plot); marker shape
# encodes the solver (circle = partitioned QAOA, triangle = PCE).
SURFACE, INK, INK_2, GRID = "#ffffff", "#0b0b0b", "#52514e", "#e4e3df"
TAB10 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
COLORS = {v: TAB10[i] for i, v in enumerate(VARIANTS)}
LABELS = {
    "qaoa_random": "QAOA Random", "qaoa_tqa": "QAOA TQA", "qaoa_linramp": "QAOA Linear ramp",
    "iter_interp": "Iterative QAOA INTERP", "iter_fourier": "Iterative QAOA FOURIER",
    "iter_chebyshev": "Iterative QAOA CHEBYSHEV",
    "pce_random": "PCE Random", "pce_progressive": "ProgressivePCE",
}


def postprocess(run_dir: Path, run_json: Path) -> int:
    import csv

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads(run_json.read_text(encoding="utf-8"))
    problems: dict[str, dict] = {}
    for name, block in (data.get("results") or {}).items():
        if not block or not (block.get("extra") or {}).get("runs"):
            continue
        m = re.match(r"^(po_a\d+_t\d+_(?:orig|s\d\d)_b\d+)_l(.+)$", name)
        if not m:
            continue
        problem, lam = m.groups()
        problems.setdefault(problem, {})[lam] = block
    if not problems:
        print("postprocess: no init_test results to plot")
        return 0

    for problem, by_lam in problems.items():
        lams = sorted(by_lam, key=float)
        table = []
        fig, ax = plt.subplots(figsize=(10, 6), facecolor=SURFACE)
        ax.set_facecolor(SURFACE)
        for variant in VARIANTS:
            marker = "o" if variant in QAOA_VARIANTS else "^"
            xs, ys, feas = [], [], []
            for lam in lams:
                runs = [r for r in by_lam[lam]["extra"]["runs"]
                        if r["variant"] == variant and r.get("objective") is not None]
                if not runs:
                    continue
                good = [r for r in runs if r["feasible"]]
                best = min(good or runs, key=lambda r: r["objective"])
                xs.append(float(lam)), ys.append(best["objective"]), feas.append(bool(good))
                bkv = (by_lam[lam].get("best_known") or {}).get("value")
                table.append({"problem": problem, "lambda": lam, "solver": best["solver"], "variant": variant,
                              "best_objective": best["objective"], "feasible": bool(good),
                              "seeds": len(runs), "best_known": bkv, "gap": _gap(best["objective"], bkv)})
            if not xs:
                continue
            c = COLORS[variant]
            ax.plot(xs, ys, color=c, lw=2, zorder=2, marker=marker, ms=8, label=LABELS[variant])
            for x, y, f in zip(xs, ys, feas):
                if not f:  # hollow marker: best run was infeasible
                    ax.plot([x], [y], marker=marker, ms=8, color=c, mfc=SURFACE, mew=2, zorder=3)
        bk = [(float(l), (by_lam[l].get("best_known") or {}).get("value")) for l in lams]
        bk = [(x, y) for x, y in bk if y is not None]
        if bk:
            ax.plot(*zip(*bk), linestyle="none", marker="_", ms=24, mew=2.5, color=INK, zorder=4,
                    label="QOBLIB best known")
        ax.set_xscale("symlog", linthresh=1e-6)
        ax.set_xticks([float(l) for l in lams])
        ax.set_xticklabels(lams, rotation=45, color=INK_2)
        ax.tick_params(colors=INK_2)
        ax.grid(True, color=GRID, lw=0.8, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.set_xlabel("risk weight λ", color=INK_2)
        ax.set_ylabel("official objective, best of seeds (lower is better)", color=INK_2)
        has_pce = any(row["variant"] in PCE_VARIANTS for row in table)
        shapes = "circle = partitioned QAOA, triangle = PCE, " if has_pce else ""
        ax.set_title(f"{problem}: objective vs λ  ({shapes}hollow = no feasible run)",
                     color=INK, loc="left", fontsize=11)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="center left", bbox_to_anchor=(1.01, 0.5))
        fig.tight_layout()
        out_dir = run_dir / problem
        out_dir.mkdir(exist_ok=True)
        png = out_dir / f"{problem}_objective_vs_lambda.png"
        fig.savefig(png, dpi=150, facecolor=SURFACE)
        plt.close(fig)
        order = {v: i for i, v in enumerate(VARIANTS)}
        table.sort(key=lambda r: (float(r["lambda"]), order[r["variant"]]))
        _write_csv(out_dir / f"{problem}_objective_vs_lambda.csv", table, POINT_FIELDS)

        # Every run behind the points (all seeds, failed runs included).
        all_runs = []
        for lam in lams:
            bkv = (by_lam[lam].get("best_known") or {}).get("value")
            for r in by_lam[lam]["extra"]["runs"]:
                all_runs.append({"problem": problem, "lambda": lam, **{k: r.get(k) for k in RUN_FIELDS[2:]},
                                 "gap": _gap(r.get("objective"), bkv)})
        all_runs.sort(key=lambda r: (float(r["lambda"]), order.get(r["variant"], 99), r["seed"]))
        _write_csv(out_dir / f"{problem}_all_runs.csv", all_runs, RUN_FIELDS)
        print(f"postprocess: wrote {png} + {problem}_objective_vs_lambda.csv + {problem}_all_runs.csv")
    return 0


POINT_FIELDS = ["problem", "lambda", "solver", "variant", "best_objective", "feasible", "seeds", "best_known", "gap"]
RUN_FIELDS = ["problem", "lambda", "solver", "variant", "seed", "objective", "feasible", "raw_best_feasible", "gap",
              "circuits", "n_qubits", "backend_time_s", "wall_s", "sol", "error"]


def _gap(objective, best_known):
    """(objective − best known) / |best known|, or None."""
    if objective is None or best_known in (None, 0):
        return None
    return (objective - best_known) / abs(best_known)


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    import csv

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--postprocess", nargs=2, metavar=("RUN_DIR", "RUN_JSON"))
    p.add_argument("--instance")
    p.add_argument("--data-dir")
    p.add_argument("--assets", type=int)
    p.add_argument("--days", type=int)
    p.add_argument("--lam")
    p.add_argument("--budget", type=int)
    p.add_argument("--cloud", default="false")
    p.add_argument("--quick", action="store_true", help="tiny settings for smoke tests")
    p.add_argument("--variants", nargs="+", choices=VARIANTS,
                   help="variants to run (default: the 6 QAOA ones; PCE only on request)")
    p.add_argument("--seeds", nargs="+", type=int, help="override the seeds (default 1 2 3)")
    args = p.parse_args(argv)
    if args.postprocess:
        return postprocess(Path(args.postprocess[0]), Path(args.postprocess[1]))
    missing = [k for k in ("instance", "data_dir", "assets", "days", "lam", "budget") if getattr(args, k) is None]
    if missing:
        p.error(f"missing arguments: {missing}")
    return run_worker(args)


if __name__ == "__main__":
    sys.exit(main())
