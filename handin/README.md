# Portfolio (QOBLIB 06) solutions to hand in

Best verified solution per instance so far, re-scored with the official referee in
`qoblib_tools.py` (all feasible). Binary variables = the official QUBO size,
`(6n + cash slack + count slack) * T`. λ = 1e-2 is left out: the pipeline is still
worse than the empty portfolio there.

| file | binary vars | score | best known | gap | method |
|---|---:|---:|---:|---:|---|
| `po_a010_t10_orig_b004_l1e-3.sol` | 710 | -8397 | -8397 | +0.00% | SA + blocks + iterated search, 20 s |
| `po_a050_t10_orig_b020_l1e-4.sol` | 3,110 | -370869 | -370869 | +0.00% | SA + blocks + iterated search, 120 s |
| `po_a050_t10_s00_b020_l1e-4.sol` | 3,110 | -355367 | -355530 | +0.05% | SA + blocks + iterated search, 120 s |
| `po_a050_t10_s01_b020_l1e-4.sol` | 3,110 | -301400 | -301400 | +0.00% | SA + blocks + iterated search, 120 s |
| `po_a050_t10_s02_b020_l1e-4.sol` | 3,110 | -443266 | -443357 | +0.02% | SA + blocks + iterated search, 120 s |
| `po_a050_t10_orig_b020_l1e-3.sol` | 3,110 | -204581 | -206239 | +0.80% | SA + blocks + iterated search, 1200 s |
| `po_a050_t10_s00_b020_l1e-3.sol` | 3,110 | -178637 | -191879 | +6.90% | SA + blocks + iterated search, 120 s |
| `po_a050_t10_s01_b020_l1e-3.sol` | 3,110 | -131333 | -141376 | +7.10% | SA + blocks + iterated search, 120 s |
| `po_a050_t10_s02_b020_l1e-3.sol` | 3,110 | -223691 | -233766 | +4.31% | SA + blocks + iterated search, 120 s |
| `po_a200_t10_orig_b050_l1e-3.sol` | 12,110 | -780355 | — | — | SA + blocks + iterated search, 300 s |
| `po_a400_t15_orig_b100_l1e-3.sol` | 36,165 | -2815258 | — | — | empty start + blocks + iterated search, 1500 s |

Method: SA on the compact QUBO (`portfolio_compact.py`) + fast repair, then exact
day × asset-group blocks (groups of 3, net-position space) and iterated local search
(`asset_groups.py pipeline`). Quantum programs so far are experiments only
(`qaoa_signed_blocks.py`: up to 24 qubits on Qoro's cloud); none of these files uses one.

Re-check a file:

```python
import qoblib_tools as qt
h, pos = qt.read_portfolio_solution("handin/po_a050_t10_orig_b020_l1e-4.sol")
m = qt.PortfolioModel(qt.load_portfolio_instance(h["instance"]), budget=int(h["budget"]), lam=h["lambda"])
print(m.score(pos))
```
