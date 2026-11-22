# Portfolio (QOBLIB 06) solutions to hand in

Best verified solution per instance so far, re-scored with the official referee in
`qoblib_tools.py` (all feasible). Binary variables = the official QUBO size,
`(6n + cash slack + count slack) * T`. λ = 1e-2 `s01` is left out: nothing better than
the empty portfolio found (-1000, best known -2248; its best-known book is 12 hedged units
on day 1 only). Negative gaps **beat** QOBLIB's best-known value: the reference files
re-score exactly to their published objectives with the same referee, so the comparison is
like for like.

| file | binary vars | score | best known | gap | method |
|---|---:|---:|---:|---:|---|
| `po_a010_t10_orig_b004_l1e-3.sol` | 710 | -8397 | -8397 | +0.00% | SA + blocks + iterated search, 20 s |
| `po_a050_t10_orig_b020_l1e-4.sol` | 3,110 | -370869 | -370869 | +0.00% | SA + blocks + iterated search, 120 s |
| `po_a050_t10_s00_b020_l1e-4.sol` | 3,110 | -355530 | -355530 | +0.00% | day_swap (warm start from the previous hand-in), 60 s |
| `po_a050_t10_s01_b020_l1e-4.sol` | 3,110 | -301400 | -301400 | +0.00% | SA + blocks + iterated search, 120 s |
| `po_a050_t10_s02_b020_l1e-4.sol` | 3,110 | -443357 | -443357 | +0.00% | day_swap (warm start from the previous hand-in), 60 s |
| `po_a050_t10_orig_b020_l1e-3.sol` | 3,110 | -206239 | -206239 | +0.00% | day_swap (warm start from the previous hand-in), 60 s |
| `po_a050_t10_s00_b020_l1e-3.sol` | 3,110 | -191879 | -191879 | +0.00% | day_swap (warm start from the previous hand-in), 41 s |
| `po_a050_t10_s01_b020_l1e-3.sol` | 3,110 | -141376 | -141376 | +0.00% | day_swap (warm start from the previous hand-in), 60 s |
| `po_a050_t10_s02_b020_l1e-3.sol` | 3,110 | -233766 | -233766 | +0.00% | day_swap (warm start from the previous hand-in), 90 s + 45 s with two-day moves |
| `po_a050_t10_orig_b020_l1e-2.sol` | 3,110 | -47301 | -37107 | **-27.47%** | day_swap (warm start from the previous hand-in), 60 s + 120 s |
| `po_a050_t10_s00_b020_l1e-2.sol` | 3,110 | -24353 | -12340 | **-97.35%** | day_swap (warm start from the previous hand-in), 60 s + 120 s |
| `po_a050_t10_s02_b020_l1e-2.sol` | 3,110 | -52358 | -43565 | **-20.18%** | day_swap (warm start from the previous hand-in), 60 s + 120 s |
| `po_a200_t10_orig_b050_l1e-3.sol` | 12,110 | -840155 | — | — | day_swap (warm start from the previous hand-in), ~30 min (1-2 processes) |
| `po_a400_t15_orig_b100_l1e-3.sol` | 36,165 | -3162386 | — | — | empty start + blocks + ILS (1500 s), then day_swap: 2 min, 3 min, 17 rounds x 6 day-shard workers x 240 s (last 3 full rounds found nothing) |

**day_swap** (`day_swap.py`, parallel driver `day_swap_parallel.py`; 2026-11-22). At scale the
unit budget binds on every day (a400: 100 of 100 units), so an improving move must take a unit
from one asset and give it to another; a random group of 3 almost never holds that pair. For
one whole day (all assets, the other days fixed) the official score is `v A v + Σ f_i(v_i)`
(closed form `day_form`, checked equal to `big_blocks.block_form`). Steepest descent applies the
best of all single value changes and all ±1 × ±1 pair moves (numba; ±2 steps are redundant
because the objective is convex along lines: PSD risk up to rounding, linear return / interest,
|Δv| trading). Per-day iterated local search on top: small kicks (2-8 random unit moves),
*ruin* (zero 10-40% of the held positions) and *shrink* (one unit off 20-60%) kicks — hedged
long/short books are deep traps (a400's last day held 83 units with no return: emptying it saves
8227, removing any one unit costs 172-2040) — threshold acceptance (working state may sit
2.5e-4 x |day cost| above the day's best), and two-day moves (copy or drop the same assets on
two neighbouring days, then alternating day descents), which found the last 80 on s02 λ1e-3.
Every accepted move is re-scored with the official referee. a400 runs as contiguous day shards
in parallel processes (6 x 240 s rounds); a shard's days are merged only if the official score
improves. Restarting an a400 day from empty lands within ~0.3% of the current day cost with
15-31 different assets, so the remaining headroom per day looks small.

Older rows: SA on the compact QUBO (`portfolio_compact.py`) + fast repair, then exact
day × asset-group blocks (groups of 3, net-position space) and iterated local search
(`asset_groups.py pipeline`). Rows marked "whole-day blocks" also re-solve every day
with all assets at once (integer SA on the block's exact form, `big_blocks.py`;
`--day-blocks`), alternating with the small blocks and with kicks; seed 0 unless the
row says otherwise. Quantum programs so far are experiments only
(`qaoa_signed_blocks.py`: up to 24 qubits on Qoro's cloud); none of these files uses one.

Re-check a file:

```python
import qoblib_tools as qt
h, pos = qt.read_portfolio_solution("handin/po_a050_t10_orig_b020_l1e-4.sol")
m = qt.PortfolioModel(qt.load_portfolio_instance(h["instance"]), budget=int(h["budget"]), lam=h["lambda"])
print(m.score(pos))
```
