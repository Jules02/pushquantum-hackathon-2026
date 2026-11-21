#!/usr/bin/env python
"""Run a solver script on many QOBLIB portfolio instances, in parallel (spec: AutoRunner.md).

Usage (the first argument is always the worker script to execute):

    python auto_runner.py WORKER.py --tests init_test --silent
    python auto_runner.py WORKER.py --tests local_size_tests --seed s01 --risk 6
    python auto_runner.py WORKER.py --manual 12 21:7          # CSV rows 12 and 21 (row 21 at λ index 7)
    python auto_runner.py WORKER.py --tests all --cloud --max-parallel 2
    python auto_runner.py WORKER.py --list                    # tests, risks and CSV rows
    python auto_runner.py WORKER.py --tests init_test --dry-run

Every task runs ``WORKER.py`` in its own process (a visible terminal window unless ``--silent``)
from a fresh results directory ``AutoData/<test>_<HHMM>/``, with:

    --instance po_a010_t10_orig_b004_l1e-4  --data-dir <relative path to instances/po_a010_t10_orig>
    --assets 10 --days 10 --lam 1e-4 --budget 4 --cloud false  [--worker-args ...]

The worker writes ``<instance>.sol`` and ``<instance>.result.json`` (``{"score", "best_sol",
"extra"}``) into its working directory. The runner checks every best ``.sol`` with
``solution_checker.py``, looks up the best-known value, and writes ``<test>_<HHMM>.json``. If the
worker accepts ``--postprocess RUN_DIR JSON`` it is called once at the end (e.g. for plots).
"""
from __future__ import annotations

import argparse
import ast
import csv
import datetime as dt
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent
AUTODATA = REPO / "AutoData"
INSTANCES_DIR = REPO / "instances"
CSV_PATH = REPO / "qoblib_instance_benchmarks.csv"
CHECKER = REPO / "solution_checker.py"
SUBMISSIONS = REPO / "submissions"

# Risk weights λ, indexed 0..7 (QOBLIB file-name spelling, as in notebook 01's ``lam_tag``).
RISKS = ["0", "1e-6", "1e-5", "5e-5", "1e-4", "5e-4", "1e-3", "1e-2"]
DEFAULT_RISK = 4  # 1e-4
SEEDS = ["orig", "s00", "s01", "s02"]

# Tests in the order of AutoRunner.md (index = position). ``{XX}`` is replaced by --seed; entries
# without ``_l…`` take the λ from --risk. The spec's typos are normalised (p0_a10 -> po_a010,
# l1e-05 -> 1e-5). ``init_test`` is the initialisation comparison (3 problems × all 8 λ).
TESTS: dict[str, list[str]] = {
    "CompareToBest": ["po_a010_t10_s02_b004_l1e-5", "po_a010_t15_{XX}_b004", "po_a050_t10_s02_b020_l0"],
    "CompareToOptimal": ["po_a010_t10_s02_b004_l1e-5", "po_a010_t15_orig_b004_l1e-6", "po_a050_t15_s02_b020_l0"],
    "local_size_tests": ["po_a010_t10_{XX}_b004", "po_a050_t10_{XX}_b020", "po_a200_t10_{XX}_b050"],
    "cloud_size_tests": ["po_a050_t15_{XX}_b020", "po_a200_t15_{XX}_b050", "po_a400_t15_{XX}_b100"],
    "local_small_risk_tests": [f"po_a010_t10_{{XX}}_b004_l{lam}" for lam in ("1e-6", "1e-4", "1e-2")],
    "local_big_risk_tests": [f"po_a050_t15_{{XX}}_b020_l{lam}" for lam in ("1e-6", "1e-4", "1e-2")],
    "cloud_small_risk_tests": [f"po_a200_t15_{{XX}}_b050_l{lam}" for lam in ("1e-6", "1e-4", "1e-2")],
    "cloud_big_risk_tests": [f"po_a400_t15_{{XX}}_b100_l{lam}" for lam in ("1e-6", "1e-4", "1e-2")],
    "init_test": [
        f"{base}_l{lam}"
        for base in ("po_a010_t10_orig_b004", "po_a010_t10_s02_b004", "po_a050_t10_orig_b020")
        for lam in RISKS
    ],
}
TEST_NAMES = list(TESTS)

NAME_RE = re.compile(r"^(?:po_|p0_)?a(\d+)_t(\d+)_(orig|s\d\d)_b(\d+)(?:_l([0-9.eE+-]+))?$")


# ── instances ──────────────────────────────────────────────────────────────────────────


def norm_lam(text: str) -> str:
    """``'1e-05'``, ``'0.0001'``, ``'1e-4'`` → the canonical spelling in RISKS."""
    value = float(text)
    for r in RISKS:
        if float(r) == value:
            return r
    raise ValueError(f"λ={text} is not one of {RISKS}")


@dataclass
class Task:
    name: str  # po_a010_t10_orig_b004_l1e-4 (the official .sol stem)
    po_base: str  # po_a010_t10_orig
    assets: int
    days: int
    seed_tag: str
    budget: int
    lam: str
    n_vars: int = 0
    est_mb: float = 0.0
    est_circuits: int = 0

    @property
    def csv_instance(self) -> str:  # a010_t10_orig_b004
        return f"{self.po_base[3:]}_b{self.budget:03d}"


def make_task(text: str, lam_default: str) -> Task:
    m = NAME_RE.match(text.strip())
    if not m:
        raise ValueError(f"cannot parse instance name {text!r}")
    a, t, seed, b, lam = m.groups()
    assets, days, budget = int(a), int(t), int(b)
    lam = norm_lam(lam) if lam else lam_default
    po_base = f"po_a{assets:03d}_t{days:02d}_{seed}"
    task = Task(f"{po_base}_b{budget:03d}_l{lam}", po_base, assets, days, seed, budget, lam)
    # Official QUBO size: 6nT position bits + (4 cash + 7 budget) slack bits per day.
    task.n_vars = 6 * assets * days + 11 * days
    # Memory: the dense intra-day risk block of the BQM dominates ((6n)² couplings per day).
    task.est_mb = 400 + 3.6e-5 * (6 * assets) ** 2 * days
    # Circuits per partitioned-QAOA run: ~n_vars/12 sub-programs × (50 × 10 + 1) circuits.
    task.est_circuits = int(round(task.n_vars / 12)) * 501
    return task


def tasks_for_test(test: str, seed: str, lam: str) -> list[Task]:
    return [make_task(entry.replace("{XX}", seed), lam) for entry in TESTS[test]]


def load_csv() -> list[dict]:
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def tasks_for_manual(entries: list[str], rows: list[dict]) -> list[Task]:
    tasks = []
    for entry in entries:
        idx, _, risk = entry.partition(":")
        row = rows[int(idx)]
        lam = RISKS[int(risk)] if risk else RISKS[DEFAULT_RISK]
        tasks.append(make_task(f"po_{row['Instance']}", lam))
    return tasks


# ── best-known values ──────────────────────────────────────────────────────────────────

_BKV_CACHE: dict[str, dict] = {}


def _import_qt():
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import qoblib_tools as qt

    return qt


def load_instance_dir(path: Path, name: str):
    """Parse ``stock_prices.txt.gz`` + ``covariance_matrices.txt.gz`` like ``qt.load_portfolio_instance``."""
    from fractions import Fraction

    qt = _import_qt()
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
    cov = {}
    for day, s1, s2, value in qt._data_rows(path / "covariance_matrices.txt.gz"):
        t = int(day)
        if s1 in index and s2 in index and t < n_periods:
            cov[index[s1], index[s2], t] = Fraction(value)
    return qt.PortfolioInstance(name, symbols, n_periods, prices, cov)


def _model(task: Task):
    qt = _import_qt()
    local = INSTANCES_DIR / task.po_base
    inst = load_instance_dir(local, task.po_base) if local.is_dir() else qt.load_portfolio_instance(task.po_base)
    return qt.PortfolioModel(inst, budget=task.budget, lam=task.lam)


def best_known(task: Task, rows: list[dict]) -> dict:
    """CSV ``Best objective`` → QOBLIB curated reference (scored) → best of ``submissions/``."""
    if task.name in _BKV_CACHE:
        return _BKV_CACHE[task.name]
    row = next((r for r in rows if r["Instance"] == task.csv_instance), None)
    out = {"value": None, "source": None, "status": row["Status"] if row else None,
           "csv_parameters": row["Parameters"] if row else None, "note": None}
    if row:
        params = dict(re.findall(r"(Assets|Periods|Budget) (\d+)", row["Parameters"]))
        expected = {"Assets": str(task.assets), "Periods": str(task.days), "Budget": str(task.budget)}
        if params != expected:
            out["note"] = f"CSV Parameters {params} differ from the instance name {expected}"
        if row["Best objective"].strip():
            out.update(value=float(row["Best objective"]), source="csv")
            if out["value"].is_integer():
                out["value"] = int(out["value"])
    if out["value"] is None:
        try:
            qt = _import_qt()
            path = qt.fetch_portfolio_reference(task.po_base, budget=task.budget, lam_tag=task.lam)
            _, positions = qt.read_portfolio_solution(path)
            out.update(value=int(_model(task).score(positions).objective),
                       source="qoblib_optimal" if ".opt." in Path(path).name else "qoblib_best_known")
        except Exception as err:  # no reference (a200/a400, toy λ) or no network
            out["note"] = f"no QOBLIB reference ({type(err).__name__})"
    if out["value"] is None and SUBMISSIONS.is_dir():
        qt = _import_qt()
        stem = task.name
        files = [p for p in SUBMISSIONS.glob(f"*/**/{stem}*.sol")]
        scores = []
        if files:
            model = _model(task)
            for p in files:
                try:
                    _, positions = qt.read_portfolio_solution(p)
                    s = model.score(positions)
                    if s.feasible:
                        scores.append((s.objective, p))
                except Exception:
                    continue
        if scores:
            value, p = min(scores, key=lambda x: x[0])
            out.update(value=int(value), source=f"submissions:{p.relative_to(REPO).as_posix()}")
    _BKV_CACHE[task.name] = out
    return out


# ── solution checker ───────────────────────────────────────────────────────────────────

SCORE_RE = re.compile(r"Your score:\s*objective=(-?\d+)\s*\((feasible|INFEASIBLE)(?:;\s*(.*))?\)\s*$", re.M)
VIOL_RE = re.compile(r"Constraint violations:\s*(\[.*\])", re.S)


def run_checker(sol: Path) -> dict:
    """Run ``solution_checker.py`` on ``sol`` and parse what it prints."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    try:
        proc = subprocess.run([sys.executable, str(CHECKER), str(sol)], cwd=REPO, env=env,
                              capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=1800)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout"}
    out = {"ok": proc.returncode == 0, "returncode": proc.returncode, "stdout": proc.stdout.strip(),
           "stderr": proc.stderr.strip()[-2000:] or None,
           "objective": None, "feasible": None, "breakdown": {}, "violations": None}
    m = SCORE_RE.search(proc.stdout)
    if m:
        out["objective"] = int(m.group(1))
        out["feasible"] = m.group(2) == "feasible"
        out["breakdown"] = {k: int(v) for k, v in re.findall(r"(\w+)=(-?\d+)", m.group(3) or "")}
    v = VIOL_RE.search(proc.stdout)
    if v:
        try:
            out["violations"] = ast.literal_eval(v.group(1).strip())
        except (ValueError, SyntaxError):
            out["violations"] = v.group(1).strip()
    return out


# ── launching ──────────────────────────────────────────────────────────────────────────


def now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def unique_path(path: Path) -> Path:
    """``path`` if free, else ``path_1``, ``path_2``, … (never overwrite)."""
    if not path.exists():
        return path
    k = 1
    while True:
        cand = path.with_name(f"{path.stem}_{k}{path.suffix}")
        if not cand.exists():
            return cand
        k += 1


def child_main(task_json: str) -> int:
    """Inside the task's terminal: run the worker, tee its output to a log, write a done file."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    spec = json.loads(Path(task_json).read_text(encoding="utf-8"))
    start, code = now(), None
    print(f"[auto_runner] {spec['name']}  started {start}\n[auto_runner] {shlex.join(spec['cmd'])}\n", flush=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    try:
        with open(spec["log"], "w", encoding="utf-8") as log:
            proc = subprocess.Popen(spec["cmd"], cwd=spec["cwd"], env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            for line in proc.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            code = proc.wait()
    except Exception as err:
        print(f"[auto_runner] failed to run the worker: {err!r}", flush=True)
        code = -1
    finally:
        done = {"exit_code": code, "start": start, "end": now()}
        tmp = Path(spec["done"] + ".tmp")
        tmp.write_text(json.dumps(done), encoding="utf-8")
        os.replace(tmp, spec["done"])
    print(f"\n[auto_runner] {spec['name']} finished with exit code {code}", flush=True)
    if not spec.get("silent"):
        time.sleep(3)  # leave the window readable for a moment
    return 0


def _visible_launcher(cmd: list[str], cwd: Path):
    """Open ``cmd`` in a new terminal window. Returns (Popen | None, tracks_child: bool)."""
    system = platform.system()
    if system == "Windows":
        return subprocess.Popen(cmd, cwd=cwd, creationflags=subprocess.CREATE_NEW_CONSOLE), True
    if system == "Darwin":
        script = f"cd {shlex.quote(str(cwd))} && {shlex.join(cmd)}"
        script = script.replace("\\", "\\\\").replace('"', '\\"')
        return subprocess.Popen(["osascript", "-e", f'tell application "Terminal" to do script "{script}"']), False
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return None, False
    for term, prefix, tracks in (("gnome-terminal", ["--wait", "--"], True),
                                 ("konsole", ["--nofork", "-e"], True),
                                 ("xterm", ["-e"], True),
                                 ("x-terminal-emulator", ["-e"], False)):
        exe = shutil.which(term)
        if exe:
            return subprocess.Popen([exe, *prefix, *cmd], cwd=cwd), tracks
    return None, False


@dataclass
class Running:
    task: Task
    proc: subprocess.Popen | None
    tracks: bool
    launched: float
    start: str
    credits_used_at_start: float | None = None
    overlapped: bool = False
    exited_at: float | None = None


@dataclass
class Scheduler:
    cloud: bool
    max_parallel: int | None
    low_credits: float
    slots: int = 1
    mem_budget_mb: float = float("inf")
    info: dict = field(default_factory=dict)

    def plan(self, tasks: list[Task]) -> None:
        if self.cloud:
            bal = credit_status()
            self.info["credits_at_start"] = bal
            balance = bal.get("balance") if bal else None
            if balance is None:
                self.slots = 1
                self.info["reason"] = "credit balance unavailable: one task at a time"
            elif balance < self.low_credits:
                self.slots = 1
                self.info["reason"] = f"balance {balance} < --low-credits {self.low_credits}: one task at a time"
            else:
                self.slots = self.max_parallel or 4
                self.info["reason"] = f"balance {balance}: up to {self.slots} tasks in parallel"
            return
        try:
            import psutil

            cores = psutil.cpu_count(logical=False) or os.cpu_count() or 2
            avail = psutil.virtual_memory().available / 2**20
        except ImportError:
            cores, avail = os.cpu_count() or 2, float("inf")
        self.slots = max(1, cores // 2)  # Maestro is multithreaded: ~2 cores per task
        if self.max_parallel:
            self.slots = min(self.slots, self.max_parallel)
        self.mem_budget_mb = 0.8 * avail
        self.info.update(physical_cores=cores, available_mb=round(avail), slots=self.slots,
                         mem_budget_mb=round(self.mem_budget_mb),
                         reason="local: cores//2 tasks, total estimated memory ≤ 80% of free RAM")

    def can_launch(self, task: Task, running: list[Running]) -> bool:
        if not running:
            return True
        if len(running) >= self.slots:
            return False
        if self.cloud:
            return True
        return sum(r.task.est_mb for r in running) + task.est_mb <= self.mem_budget_mb


def credit_status() -> dict | None:
    try:
        from divi.backends import QoroService

        raw = QoroService().get_credit_balance()
        return {"balance": float(raw.get("balance")), "total_used": float(raw.get("total_used", 0) or 0)}
    except Exception as err:
        print(f"[auto_runner] could not read the Qoro credit balance: {err}")
        return None


# ── one test ───────────────────────────────────────────────────────────────────────────


def run_test(test: str, tasks: list[Task], args, rows: list[dict]) -> Path | None:
    worker = Path(args.worker).resolve()
    sched = Scheduler(cloud=args.cloud, max_parallel=args.max_parallel, low_credits=args.low_credits)
    sched.plan(tasks)

    print(f"\n=== {test}: {len(tasks)} task(s), {'cloud' if args.cloud else 'local'}, "
          f"{'silent' if args.silent else 'terminal windows'} ===")
    print(f"scheduler: {sched.info}")
    for t in tasks:
        data = INSTANCES_DIR / t.po_base
        print(f"  {t.name:32s} vars={t.n_vars:6d}  est_mem≈{t.est_mb:7.0f} MB  "
              f"circuits/QAOA run≈{t.est_circuits:,}  data={'ok' if data.is_dir() else 'MISSING (download)'}")
    if args.dry_run:
        for t in tasks:
            print(f"  best known {t.name}: {best_known(t, rows)}")
        return None

    AUTODATA.mkdir(exist_ok=True)
    run_dir = unique_path(AUTODATA / f"{test}_{dt.datetime.now():%H%M}")
    run_dir.mkdir(parents=True)
    tasks_dir = run_dir / "_tasks"
    tasks_dir.mkdir()
    print(f"results directory: {run_dir}")

    silent = args.silent
    pending, running, finished = list(tasks), [], {}
    extra_args = shlex.split(args.worker_args) if args.worker_args else []
    while pending or running:
        # Collect finished tasks.
        for r in list(running):
            done = run_dir / f"{r.task.name}.done.json"
            if done.exists():
                finished[r.task.name] = finalize(r, json.loads(done.read_text(encoding="utf-8")), run_dir, args, rows)
                running.remove(r)
            elif r.tracks and r.proc is not None and r.proc.poll() is not None:
                r.exited_at = r.exited_at or time.time()
                if time.time() - r.exited_at > 10:  # terminal closed without a done file
                    finished[r.task.name] = finalize(r, {"exit_code": None, "start": r.start, "end": now(),
                                                         "error": "process exited without a done file"},
                                                     run_dir, args, rows)
                    running.remove(r)
        # Launch new ones.
        while pending and sched.can_launch(pending[0], running):
            task = pending.pop(0)
            data_rel = os.path.relpath(INSTANCES_DIR / task.po_base, run_dir)
            worker_cmd = [sys.executable, str(worker), "--instance", task.name, "--data-dir", data_rel,
                          "--assets", str(task.assets), "--days", str(task.days), "--lam", task.lam,
                          "--budget", str(task.budget), "--cloud", "true" if args.cloud else "false",
                          *extra_args]
            spec = {"name": task.name, "cmd": worker_cmd, "cwd": str(run_dir), "silent": silent,
                    "log": str(run_dir / f"{task.name}.log"), "done": str(run_dir / f"{task.name}.done.json")}
            spec_path = tasks_dir / f"{task.name}.json"
            spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
            child = [sys.executable, str(Path(__file__).resolve()), "--_child", str(spec_path)]
            proc, tracks = None, True
            if not silent:
                proc, tracks = _visible_launcher(child, run_dir)
                if proc is None:
                    print("[auto_runner] no terminal emulator / display found: falling back to --silent")
                    silent = True
            if silent:
                proc = subprocess.Popen(child, cwd=run_dir, stdin=subprocess.DEVNULL,
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            used = None
            if args.cloud:
                st = credit_status()
                used = st["total_used"] if st else None
            r = Running(task, proc, tracks, time.time(), now(), used)
            if running:
                r.overlapped = True
                for other in running:
                    other.overlapped = True
            running.append(r)
            print(f"[{now()}] launched {task.name}  ({len(running)} running, {len(pending)} queued)")
        time.sleep(2)

    # Final JSON: TYPE_HHMM.json, never overwritten.
    out = {
        "test": test,
        "instances": [t.name for t in tasks],
        "settings": {
            "worker": str(worker), "cloud": args.cloud, "silent": args.silent, "seed": args.seed,
            "risk": RISKS[args.risk], "manual": args.manual, "worker_args": args.worker_args,
            "python": sys.executable, "platform": platform.platform(), "divi": _divi_version(),
        },
        "results": {name: finished.get(name) for name in (t.name for t in tasks)},
    }
    if args.cloud:
        end = credit_status()
        start = sched.info.get("credits_at_start")
        out["settings"]["credits_used_total"] = (
            end["total_used"] - start["total_used"] if end and start else None)
    out["extra"] = {"scheduler": sched.info, "postprocess": None}
    json_path = unique_path(run_dir / f"{test}_{dt.datetime.now():%H%M}.json")
    json_path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {json_path}")

    # Optional worker post-processing (plots); its outcome goes into "extra".
    pp = subprocess.run([sys.executable, str(worker), "--postprocess", str(run_dir), str(json_path)],
                        cwd=run_dir, capture_output=True, text=True, encoding="utf-8", errors="replace",
                        env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    out["extra"]["postprocess"] = {"returncode": pp.returncode,
                                   "output": (pp.stdout + pp.stderr).strip()[-3000:]}
    if pp.returncode != 0:
        print(f"[auto_runner] worker post-processing unavailable or failed (exit {pp.returncode})")
    else:
        print(pp.stdout.strip())
    json_path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    _summary(out)
    return json_path


def finalize(r: Running, done: dict, run_dir: Path, args, rows: list[dict]) -> dict:
    task = r.task
    block = {"start": done.get("start", r.start), "end": done.get("end", now()),
             "exit_code": done.get("exit_code"), "error": done.get("error"),
             "parameters": {"assets": task.assets, "days": task.days, "lambda": task.lam,
                            "budget": task.budget, "seed": task.seed_tag, "n_vars": task.n_vars},
             "credits_used": None, "best_sol": None, "score": None, "checker": None,
             "best_known": None, "gap": None}
    if args.cloud:
        st = credit_status()
        if st and r.credits_used_at_start is not None:
            block["credits_used"] = {"value": st["total_used"] - r.credits_used_at_start,
                                     "approximate": r.overlapped}
    result_path = run_dir / f"{task.name}.result.json"
    worker_extra = None
    if result_path.exists():
        try:
            res = json.loads(result_path.read_text(encoding="utf-8"))
            block["best_sol"], block["score"] = res.get("best_sol"), res.get("score")
            worker_extra = res.get("extra")
        except json.JSONDecodeError as err:
            block["error"] = f"unreadable result file: {err}"
    else:
        block["error"] = block["error"] or "worker wrote no result file"
    if block["best_sol"]:
        sol = (run_dir / block["best_sol"]).resolve()
        block["checker"] = run_checker(sol) if sol.exists() else {"ok": False, "error": f"{sol} missing"}
    block["best_known"] = best_known(task, rows)
    bkv = block["best_known"]["value"]
    obj = (block["checker"] or {}).get("objective")
    if obj is None:
        obj = block["score"]
    if bkv not in (None, 0) and obj is not None:
        block["gap"] = (obj - bkv) / abs(bkv)
    block["extra"] = worker_extra
    print(f"[{now()}] finished {task.name}: exit={block['exit_code']} score={block['score']} "
          f"best_known={bkv} gap={'' if block['gap'] is None else f'{block['gap']:+.2%}'}")
    return block


def _divi_version() -> str | None:
    try:
        from importlib.metadata import version

        return version("qoro-divi")
    except Exception:
        return None


def _summary(out: dict) -> None:
    print(f"\n{'instance':34s} {'score':>12s} {'best known':>12s} {'gap':>9s}  feasible")
    for name, b in out["results"].items():
        if not b:
            print(f"{name:34s}  (no result)")
            continue
        chk = b.get("checker") or {}
        gap = "" if b.get("gap") is None else f"{b['gap']:+.2%}"
        bk = (b.get("best_known") or {}).get("value")
        print(f"{name:34s} {str(b.get('score')):>12s} {str(bk):>12s} {gap:>9s}  {chk.get('feasible')}")


# ── CLI ────────────────────────────────────────────────────────────────────────────────


def resolve_tests(items: list[str]) -> list[str]:
    if any(i.lower() == "all" for i in items):
        return list(TEST_NAMES)
    lower = {n.lower(): n for n in TEST_NAMES}
    out = []
    for item in items:
        if item.isdigit():
            out.append(TEST_NAMES[int(item)])
        elif item.lower() in lower:
            out.append(lower[item.lower()])
        else:
            raise SystemExit(f"unknown test {item!r}; choose from {TEST_NAMES} (or indices 0-{len(TEST_NAMES) - 1})")
    return out


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["--_child"]:
        return child_main(argv[1])

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("worker", help="python file executed once per instance")
    p.add_argument("--tests", nargs="+", default=None, help=f"test names or indices: {TEST_NAMES}, or 'all'")
    p.add_argument("--manual", nargs="+", default=None, metavar="ROW[:RISK]",
                   help="0-based rows of qoblib_instance_benchmarks.csv (overrides tests/seed/risk)")
    p.add_argument("--seed", default="orig", choices=SEEDS, help="data seed used for XX (default orig)")
    p.add_argument("--risk", type=int, default=DEFAULT_RISK, choices=range(len(RISKS)),
                   help=f"λ index into {RISKS} (default {DEFAULT_RISK} = {RISKS[DEFAULT_RISK]})")
    p.add_argument("--cloud", action="store_true", help="run on Qoro's cloud (uses credits); default local")
    p.add_argument("--silent", action="store_true", help="no terminal windows")
    p.add_argument("--max-parallel", type=int, default=None, help="cap on simultaneous tasks")
    p.add_argument("--low-credits", type=float, default=100.0, help="below this balance: one task at a time")
    p.add_argument("--worker-args", default="", help="extra arguments passed verbatim to the worker")
    p.add_argument("--dry-run", action="store_true", help="show the plan and best-known values, launch nothing")
    p.add_argument("--list", action="store_true", help="list tests, risks and CSV rows")
    args = p.parse_args(argv)

    rows = load_csv()
    if args.list:
        print("Tests:")
        for i, n in enumerate(TEST_NAMES):
            print(f"  {i}: {n:24s} {', '.join(TESTS[n])}")
        print("\nRisks (λ): " + ", ".join(f"{i}={r}" for i, r in enumerate(RISKS)))
        print("\nCSV rows (for --manual):")
        for i, r in enumerate(rows):
            print(f"  {i:2d}: {r['Instance']:22s} {r['Parameters']:40s} {r['Status']}")
        return 0

    if not Path(args.worker).is_file():
        raise SystemExit(f"worker script {args.worker!r} not found")
    if args.manual:
        plan = [("manual", tasks_for_manual(args.manual, rows))]
    else:
        tests = resolve_tests(args.tests or ["init_test"])
        plan = [(t, tasks_for_test(t, args.seed, RISKS[args.risk])) for t in tests]
    for test, tasks in plan:
        run_test(test, tasks, args, rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
