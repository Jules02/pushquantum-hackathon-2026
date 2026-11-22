# Where quantum fits in our portfolio pipeline

**In short:** we built a hybrid pipeline, as the challenge kit suggests ("Hybrid loops", notebook 01 §1.8).
Classical search finds the global solution. A QAOA block solver re-solves small asset × day windows of
that solution and keeps a change only when the official score improves. We report each stage's effect
separately, as the README asks: *"compare every quantum result with the classical baselines before
claiming a quantum contribution."*

## What produces the scores

Classical methods produce every score in `README.md`:
SA on a compact QUBO → repair → exact day × asset-group blocks → whole-day blocks → iterated local search.
Instances go up to 36,165 binary variables (a400_t15), with gaps of 0–1.5% at λ ≤ 1e-3 on a050.

## What the quantum part does, and what it gave

- **An encoding that fits QAOA.** Each asset's net position takes 3 qubits (half of unary's 6). Every
  bitstring is a valid holding, and the block QUBO is exact.
- **QAOA as a block solver (9 qubits, TQA start, local Maestro).** On a010 λ1e-3 the pipeline with QAOA
  as its only block solver reaches the best-known value, −8397. On a050 its blocks matched the exact
  solver, but it ran far fewer moves in the same time (+2.78% in 600 s).
- **Better than random sampling.** With the top 64 samples, QAOA finds the block optimum on 60/60
  blocks at 9 qubits, against 27/60 for 64 random samples. At 12 qubits it is 59–60/60 against 9/60.
  Starting from TQA angles finds the optimum on 6/8 blocks, against 4/8 for random angles.
- **Largest batched cloud run:** 100 QAOA programs in one ensemble on a400_t15. Details below.
- **Largest quantum program:** 24 qubits (8 assets × 3) on Qoro's cloud. Its best sample was +4293 above
  the optimum, against +16856 for random sampling.

## What didn't work

- **PCE.** It failed on the unary encoding and on whole 50-asset days (150 variables on 17 qubits): its
  best feasible day was worse than an empty one.
- **Blocks that fit a GPU (18–36 qubits) barely improve a polished solution.** Only whole-day blocks,
  50 assets or about 150 qubits, gave real gains, and classical SA solves those.
- **Cloud latency with COBYLA.** COBYLA waits for each result before its next step, so every
  iteration is a cloud round trip of about 20 s. The block loop therefore runs locally.

## Batched QAOA at full scale on Qoro's cloud (a400_t15)

We tested whether Qoro's batching makes the cloud worth using for the block loop. One Monte Carlo
QAOA step (population 30, 6 iterations, TQA start) ran on N blocks of 3 assets × 1 day (9 qubits).
The blocks came from the a400_t15 solution, which was −2,815,258 at the time; the hand-in file has
since improved. All N programs went into one `ProgramEnsemble`, so each optimiser step is one merged
cloud job. Script: `cloud_probe.py`. Raw numbers: `results/cloud_probe/probe.jsonl`.

| Backend | N=1 | N=10 | N=100 | Credits, N=100 | Block optimum found, N=100 |
|---|---|---|---|---|---|
| local Maestro | – | 0.9 s | – | – | 10/10 (N=10) |
| `qoro_cpu` | 32 s | 62 s | 362 s | 3.3 | 99/100 |
| `qoro_maestro` | 35 s | 56 s | 373 s | 3.4 | 98/100 |
| `qoro_gpu` | 366 s* | 114 s | 483 s | 4.8 | 99/100 |

\*Probably a cold start or queue wait.

- **Batching works as advertised.** Every run took 7 cloud submissions, whatever N was.
- **Run time still grows with N.** At N=100, sending the circuits took 80–100 s and fetching the
  results took about 150 s. That is about 3.6 s per block on the cloud, against about 0.09 s locally.
  GPUs don't help at 9 qubits.
- **Cost is not the limit.** About 0.034 credits per block (0.048 on GPU). A full pass over a400_t15
  (about 2000 blocks) would cost about 70 credits and take about 2 hours, against about 3 minutes
  locally.
- **No objective change.** In all 130 sampled blocks the classical solution was already the block
  optimum, so this step could not improve the score.

**Conclusion:** batched QAOA runs on Qoro's cloud at the full 400-asset × 15-day scale. It finds the
block optimum 98–99% of the time, for about 3.6 s and 0.034 credits per block. For blocks this small
the local simulator is about 40× faster, and the classical solution leaves no room to improve at this
block size. So the cloud adds no score improvement here, and our hand-in scores remain classical.
