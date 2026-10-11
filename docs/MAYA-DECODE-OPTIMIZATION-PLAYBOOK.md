# Maya / Strata decode-speed optimization playbook (VRAM + RAM + SSD tiered MoE)

Written 2026-10-11 after one experiment campaign on `project-maya` (a fork of Strata) running GLM-5.3-Flash
UD-IQ3_XXS. This file explains how to look for speed-ups in this kind of engine, not just what changed. It covers how
the bottleneck was found, how candidate ideas were chosen, how each was tested fairly, and why some were kept and
others rejected. A future agent should be able to repeat the method on a different model, machine or engine version.

---

## 0. TL;DR of results (same prompt, greedy decode, one binary, arms differ only by env switches)

| Change | Kind | Decode speed | Verdict |
|---|---|---|---|
| Baseline (as found, incl. the uncommitted static host cache) | - | **5.65 tok/s** (5.58-5.74, 6 runs) | reference |
| **Split MoE** (`STRATA_GLM_SPLIT_MOE=1`) | exact | **6.5 tok/s, +15%** (6.40-6.61, 7 runs) | **keep** |
| + learned host cache v2 (`STRATA_GLM_HOST_LEARN=2`) | exact | 6.99 tok/s mean (7.17, 6.80), **~+7% over split** | promising, needs more runs; ~1/3 of the simulated potential |
| + learned host cache v1 (`=1`) | exact | 6.51-6.60 | no gain (explained below) |
| Disk lane v1 / v2 (CPU computes SSD-read experts) | exact | 5.06 / 6.32 | **rejected** |
| SSD lookahead reads (`STRATA_GLM_AHEAD=1/2`) | exact | 6.34 / 6.26 | **rejected** |
| Next-layer RAM->VRAM prefetch (`PREFETCH_N=1`, 3 tunings) | exact | 6.29-6.30 | **rejected** |
| Bigger VRAM pool (13 -> 14.5 GB) | exact | 6.53 (no change) | not worth trading quality for VRAM |
| **Tail skip, rank >= 4** (`STRATA_GLM_TAIL_SKIP=4`) | approximate | **11.86 / 11.81 tok/s, 2.07x baseline** (2 full runs) | **best trade-off**, code still correct |
| Tail skip rank >= 4 + learned host cache v2 | approximate | 11.99 tok/s (1 run) | within noise of tail 4 alone, code correct |
| Tail skip rank >= 6 / 5 / 3 | approximate | 8.63 / 9.86 / 13.74 tok/s | frontier, see section 6 |
| Cache-aware substitution (`STRATA_GLM_SUB_FROM/SUB_M`) | approximate | slower and worse KL than plain skipping | **rejected** |
| Learned host cache v3 (`STRATA_GLM_HOST_LEARN=3`), round 2 | exact | 11.29 vs v2's 11.63 tok/s (-3%) | **rejected** (section 6b) |
| Prompt-informed LFU seeding (`STRATA_GLM_PROMPT_LFU=32/64`), round 2 | exact | 10.55-11.11 vs 10.80-11.35 tok/s | **rejected** (section 6b) |

Every tail-skip arm's generated Dijkstra passed the functional test. The repeat rank-4 run's answer was longer (it
drew an ASCII diagram of the example graph) and hit the 1,024-token cap mid-example. Its completed functions passed
the same test, checked with `results/fc-split-tail4-trimmed.json`.

---

## 1. Where everything is

- Repo: `C:\Users\Admin\Desktop\AI\project-maya`. Engine core: `src/core/glm_fast_path.cu` (tiers, the service
  thread, the boundary, fast_moe), `src/kernels/cuda/glm_fast.cu` (the route kernel, moe_wait/fetch/gate_up/down,
  also compiled for HIP), and `src/core/glm_fast_state.hpp` (FastState).
- Experiment folder: `C:\Users\Admin\Desktop\AI\maya-exp`
  - `bench.py`: the speed harness (section 3)
  - `score.py`: the teacher-forced quality harness (section 3)
  - `check_code.py`: the functional test of the generated code
  - `route_analysis.py`: routing statistics from a route log
  - `tier_sim.py`: the three-tier cache simulator over a route log
  - `alloc_sim.py`: per-layer VRAM slot allocation simulation
  - `patches/*.py`: the exact source edits for each experiment
  - `queue*.sh`: the batches that were run
  - `results/`: every log, JSON, `*.out` summary, the route log `routes.0`, and the reference dump
- Git branches, stacked in this order and all env-gated so that one binary can run every arm:
  - `exp/baseline`: the user's previously uncommitted host-cache work, committed as found
  - `exp/split-moe`
  - `exp/disk-lane`
  - `exp/tail-skip` (also adds routing weights to `STRATA_GLM_ROUTE_LOG`)
  - `exp/cache-route`
  - `exp/disk-lane2`
  - `exp/host-learn` (learned host cache v1, then v2)
- Binaries: `build-hip/strata-{split,dlane,tail,exp,exp2,exp3,exp4}.exe`. `strata-exp4.exe` contains everything.
- Build on Windows: from PowerShell, `cmd /c "...\project-maya\build-maya-hip.bat"`. Bash's `cmd` loses the working
  directory. Incremental builds take a few minutes. Copy the exe aside after each build, because a running benchmark
  locks `strata.exe`.

### Large files you may want to delete
- `project-maya/build-hip/strata-*.exe`: 7 files of about 59 MB each, about 400 MB in total.
- `maya-exp/results/ref-logsoftmax.bin`: about 230 MB, the reference for `score.py --ref`. Keep it if you will score
  again with the same prompt.
- `maya-exp/results/routes.0`: about 2 MB.

The logs are small.

---

## 2. Understand the machine, the model and the engine first

### Hardware
- GPU: RX 7900 XTX, 24 GB (HIP / ROCm 7.2 on Windows)
- CPU: Ryzen 9 7900X, 12 cores
- RAM: 64 GB DDR5-5600
- SSD: WD SN850X, PCIe 4, about 7 GB/s

### Model (GLM-5.3-Flash)
- 45 layers. Layers 0-2 are dense; the 42 MoE layers each have 288 routed experts plus 1 shared expert, top-8 routing.
- Routing uses a sigmoid router and `noaux_tc`. The selection order is the biased score. The weights are
  renormalised over the selected 8 and scaled by 2.5.
- One expert (gate+up IQ2_S, down IQ3_S) is about **9 MB**. All experts together are about **110 GB**.
- Dense weights in VRAM are about 7 GB, mostly Q6_K.

### Per token
- 336 expert uses, about 3 GB of expert weights touched. Compute is tiny next to the bytes that must be moved.

### Engine tiers (exclusive)
- **VRAM pool:** per-layer slots, live LFU with aged counts, seeded from the persistent per-machine usage profile.
- **Pinned RAM tier:** the GPU pulls from it directly over PCIe with a zero-copy kernel.
- **Pageable host cache:** the uncommitted work found in the tree. It is static, filled once at warm-up.
- **SSD:** unbuffered reads.

### How a token flows
1. The device's route kernel decides, for each expert, whether it is resident, should be fetched from RAM, is
   missing (disk), or goes to the CPU lane.
2. It publishes a request into a host-mapped ring.
3. A host **service thread** reads the disk-only experts into RAM slots and answers.
4. The device spins in `moe_wait` until the answer arrives, then fetches over PCIe, computes, and lets the CPU lane
   add its part.
5. Between tokens, `fast_boundary` applies the tier moves: promotions, demotions and spare refills.

**Lesson:** read `docs/GLM5-FLASH.md`, the in-repo "tried and not kept" notes (`bench/results/PROGRESS.md`), and
`fast_moe` / `fast_service` / `fast_boundary` before proposing anything. Half of the "obvious" ideas from the
literature had already been tried and measured there. Next-layer prefetch, LRU/LFU/learned eviction, and MTP
speculation on one GPU all lost.

---

## 3. Build a fair, cheap harness before changing anything

1. **One fixed prompt, a coding task, greedy decoding.** The prompt asks for Dijkstra's algorithm in Python using
   heapq, with type hints, path reconstruction and an example, in one markdown code block.
   - Same warm-up answer before it.
   - 384 tokens for quick runs, 1024 for full answers.
2. **Fresh engine process per run.** Restore a **frozen copy of `expert_usage.txt`** before and after each run. The
   engine rewrites this profile after every request and the warm-up fills the tiers from it, so without this the
   runs drift.
3. **Pin everything that is measured or sized at start:**
   - `STRATA_GLM_POOL_GB=13` and `STRATA_GLM_HOST_CACHE_GB=16`. The desktop's VRAM use changes over time, and an
     unpinned pool once grew until the load failed with "warmed expert tables did not upload". HIP's
     `AMD_LOG_LEVEL=1` showed the real error, "Video memory allocation failed".
   - `STRATA_GLM_CPU_PLAN=000111122`, so the CPU lane's split doesn't depend on a start-up calibration.
4. **Run every arm from one binary and toggle features with env switches.** Every new code path is behind an env
   switch whose default is the old behaviour, so the control arm is literally the same build.
5. **Know the noise.**
   - Greedy output is *not* bit-reproducible across starts. Tier timing decides whether the CPU or the GPU computes
     an expert, and the two round differently.
   - Speed noise is about ±1.5%. Use two or more runs for anything under about 5%.
6. **Measure quality separately, by teacher forcing** (`STRATA_GLM_SCORE=<prompt len>` in non-serve mode).
   - Score the baseline's own answer through the one-token decode path.
   - Dump the baseline's log-softmax once (`STRATA_GLM_SCORE_DUMP`), then compare each arm
     (`STRATA_GLM_SCORE_REF`): KL, same-top-token %, NLL, plus ms/token on an *identical* token stream.
   - **Baseline against itself gives KL 0.003-0.005, which is the noise floor.**
7. **Check function, not just KL.** `check_code.py` extracts the generated code, runs it, and tests its Dijkstra on a
   fixed graph. Note that 384 tokens is too short to finish the answer, so use 1024.
8. **Profile once with the built-in tools:**
   - `STRATA_GLM_PROF=16`: per-stage GPU stream time.
   - `STRATA_GLM_TIMING=1`: the dense weights' VRAM by kind.
   - `STRATA_GLM_TIER_DIAG=1`: where disk reads come from.
   - `STRATA_GLM_ROUTE_LOG=<file>`: every route, with ids, tier masks and weights after the `exp/tail-skip` patch.

---

## 4. Find the bottleneck, and keep re-finding it

Baseline profile, about 183 ms per token:

| Stage | ms/token | Share |
|---|---|---|
| `moe_wait` (the GPU parked on disk reads) | ~101 | 55% |
| `moe_fetch` (PCIe pulls) | ~56 | 31% |
| All compute | ~25 | 14% |

Arithmetic that made it concrete:
- About 165 experts per token cross PCIe, times 9 MB, in 56 ms. That is about 26.5 GB/s, **a saturated PCIe 4 x16
  link**.
- About 62 real SSD reads per token. The host cache served only 22% of "disk" reads, measured with an added
  host-cache hit counter. During waits that is about 7 GB/s, **a saturated SSD**.
- **The two saturated resources ran one after the other within each layer.** The device waited for the disk, then
  fetched over PCIe, then computed.

Re-profile after every kept change, because the bottleneck moves:
- **After split MoE:** RAM-tier fetches hide behind the disk wait. What remains is SSD wait (about 75 ms) plus the
  disk experts' own fetch.
- **Consequence:** more VRAM or faster PCIe barely helps any more. Only fewer SSD reads help. The pool-size sweep
  confirmed this: 11 GB gave 6.14 tok/s, 13 GB 6.56 and 14.5 GB 6.53. VRAM hit rose 43.5% -> 45.1% with no speed gain.
  This one measurement killed several ideas: re-quantizing the dense weights to free VRAM, and per-layer slot
  weighting, which was simulated at only +0.5 pt coverage.

**Principle (the user's framing, borne out by the data):** keep the decode on the GPU and VRAM, use RAM next and the
SSD last. Every optimization should either overlap a slow tier with GPU work or stop needing the slow tier at all.
Moving GPU work onto the CPU only helped where the CPU's work was *off* the critical path. The existing CPU lane does
that. The disk lane did not.

---

## 5. Exact (lossless) changes: reasoning, test, outcome

### 5.1 Split MoE: kept, +15%
- **Observation:** in `fast_moe`, `moe_wait` (disk) ran before `moe_fetch` and before any expert compute, so the GPU
  sat idle while the SSD read, and only then used PCIe.
- **Idea:** compute the experts that are already available (VRAM residents and RAM-tier fetches) while the host
  reads the disk-only ones, then run a second pass for the disk-only experts and add.
- **Implementation** (`exp/split-moe`):
  1. The route kernel writes the miss mask to device memory.
  2. A tiny `moe_split` kernel, phase 0, holds back the disk entries' plan slots so the expert kernels skip them.
  3. Run fetch, gate_up and down for everything else.
  4. Then `moe_wait`.
  5. Phase 1 restores the disk entries and nulls the others.
  6. Run a second fetch, gate_up and down with `sh_out = first result`, then swap the output buffers.
- **Why it's safe:** the expert kernels already skipped null plan entries, and `moe_down` already computes
  `out = sum + sh_out`. Only the floating-point summation order changes.
- **Measured:**
  - About 6.5 tok/s against 5.65; the wait fell from 101 to 75 ms.
  - KL 0.006-0.010, against a noise floor of 0.003-0.005. The excess is consistent with rounding-order changes
    compounding through the linear-attention state, not a semantic change.
  - The code it wrote is correct.

### 5.2 Disk lane: rejected, an instructive failure
- **Idea:** the SSD read lands in RAM anyway, so let the CPU compute it (about 0.2 ms) instead of a second 9 MB PCIe
  copy (`moe_fetch2`, about 30 ms per token).
- **v1:** didn't promote those experts into VRAM. Disk reads rose from 80 to 132 per token and the VRAM hit rate fell
  from 42% to 36%, ending at 5.06 tok/s.
  - **Lesson:** a cache that stops admitting misses stops learning. The engine's promotions are how it adapts, and
    any bypass has to keep feeding the cache.
- **v2:** kept the promotion by copying on a side stream, off the critical path. Hit rates were restored, but speed
  was 6.32 vs 6.5.
  - The CPU's compute (about 22 ms per token) now sat *inside* the disk wait.
  - The side copies competed with the next layers' fetches.
  - After split MoE, PCIe was no longer the constraint, so this traded a hidden cost for a visible one.

### 5.3 Lookahead SSD reads and next-layer prefetch: rejected again
- Re-tested because split MoE leaves the SSD and the link idle part of the time.
- `AHEAD=1/2`: foreground SSD reads fell 78 -> 72, but the background reads slowed the foreground ones, giving
  6.34 / 6.26 tok/s.
- `PREFETCH_N=1`: VRAM hit rose 43.5% -> 47%, but speed was 6.29-6.30 across three tunings: default, 4 copy blocks,
  and issue after fetch.
- **Lesson:** only an idle-only, cancellable, low-priority I/O queue could make speculative reads pay, and routing
  analysis says the payoff ceiling is small (see 5.5).

### 5.4 Learned host cache: promising, v2 about +7%
- **Observation:** the 16 GB pageable host cache is filled once and never changes. Meanwhile about 62 SSD reads per
  token land in the pinned RAM tier and evict its least-used experts straight back to "disk only". Two adaptive tiers
  sat above a frozen third one.
- **Check the idea offline first** with `tier_sim.py`, a replay of the recorded route log through a model of the
  engine's VRAM/RAM/host policies:
  - The simulator reproduced reality: 42.2% VRAM hit and 60.8 SSD reads per token, against 42.5% and about 62
    measured.
  - Its counterfactual: a learned demotion chain, or equivalently one unified LFU over pinned and pageable RAM,
    gives **60.8 -> 40 SSD reads per token (-34%)**.
- **Constraint that shapes the design:**
  - The GPU can only pull from pinned memory, and Windows pins at most about half of RAM.
  - Copying each RAM victim down at the boundary would cost about 600 MB of memcpy per token on the critical path.
- **Design** (`exp/host-learn`):
  1. Host slots get a per-(layer, expert) table.
  2. A **background spill thread** shadows the RAM tier's coldest unprotected experts into the host slots that matter
     least: free ones, ones whose expert is in VRAM or RAM anyway, then the coldest. The RAM copy stays live.
  3. The boundary pauses the thread *before* taking the tier lock, which avoids a deadlock. The pause/busy flags use
     seq_cst ordering.
  4. Misses served from a host slot use a parallel memcpy instead of an SSD read. A hit moves the expert up, and its
     host slot frees, keeping the levels exclusive.
- **v1:** host hits rose from 22% to 25% of disk-path reads, with no speed gain.
  - The tier diagnostics showed why: the engine **drops a VRAM victim straight to disk** when it's colder than the
    RAM victim, so that expert never enters RAM and is never shadowed. The simulator had sent it down a level.
- **v2:** when the RAM victim already has a host shadow, demote the VRAM victim into RAM anyway, because nothing is
  lost. This gives a VRAM -> RAM -> host cascade. Result: 7.17 and 6.80 tok/s, against about 6.5, about +7%.
- **Next step:** replace the shadowing with a true single LFU pool over pinned and pageable slots. Disk reads land in
  pinned slots, a background spill keeps pinned slots free, and pageable hits are staged by memcpy. That should reach
  the simulated 34% SSD-read cut, about +15-20%.

### 5.5 Routing analysis: why some ideas don't pay
Measured on 460 tokens of route log with `route_analysis.py`:
- **Cross-layer co-activation (the "expert pairs" idea, learned online)** predicts only 2.5% / 13% of the next
  layer's *disk* misses with 8 / 32 guesses. Misses are, almost by definition, the unpredictable experts. This is
  why predictive disk prefetch keeps losing on this model.
- **Temporal reuse:** 26.7% of a layer's experts repeat from the previous token.
- **The tail is mostly disk:** the share of picks that are disk-only grows with route rank, from 17.5% at rank 0 to
  35% at rank 7. Disk-only picks carry about 25% of the routing weight. Rank 7's mean weight is 0.17 out of 2.5.

### 5.6 What already exists, so don't rebuild it
- **Persistent usage profile:** `expert_usage.txt` orders the warm-up and seeds the LFU counts.
- **The prompt feeds the counts:** prefill adds its routing counts into the LFU.
- **Per-layer weighted slot allocation** exists but needs `pack/expert_counts.txt`, which this pack lacks. It's worth
  only about +0.5 pt here.
- **Background promotions** run at the boundary, at most 8 per token.
- **Untested idea:** a full re-tier right after a long prompt, while the link and SSD are idle. My benchmark's prompt
  is short, so this needs a long-prompt benchmark.

---

## 6. Approximate changes (quality traded for speed)

### 6.1 Tail skip of disk-only experts: the best trade-off
- **Source:** upstream Strata's `STRATA_ROUTE_TAIL_SKIP`, which reported +13-16% at KL 0.003-0.006 with HumanEval
  unchanged. Maya only had it for the MTP draft (`STRATA_GLM_MTP_KEEP`).
- **Implementation** (`exp/tail-skip`): in `moe_route_kernel`, a pick that is *disk-only* (neither in the VRAM table
  nor the RAM table) at route rank >= R is left out. Its plan entry is null, it isn't read or waited on, and the
  weights are not renormalised. A weight-threshold variant exists too: `STRATA_GLM_TAIL_SKIP_W`.
- **Results**, against teacher-forced baseline speed of about 184 ms/token:

| Rank >= | Teacher-forced | KL | Same top | Full answer | Code OK |
|---|---|---|---|---|---|
| 6 | 124 ms (1.48x) | 0.014 | 97.9% | 8.63 tok/s | yes |
| 5 | 105 ms (1.75x) | 0.022 | 97.1% | 9.86 tok/s | yes |
| **4** | **91 ms (2.03x)** | **0.019** | 97.7% | **11.86 / 11.81 tok/s** | **yes** |
| 4 + learned host v2 | 88 ms (2.09x) | 0.022 | 98.4% | 11.99 tok/s | yes |
| 3 | 76 ms (2.43x) | 0.035 | 96.6% | 13.74 tok/s | yes |

- **A second-order win:** skipped tail experts stop churning the cache. With rank >= 4, the VRAM hit rate rose from
  43% to 60% and SSD reads fell from about 78 to 28 per token.
- **Weight thresholds are worse than rank thresholds.** `TAIL_SKIP_W=0.10` barely helped. `0.20` produced an outlier
  position with KL 3.3.
- **Caveat:** this is one prompt and one functional test. Before making it a default, run a coding benchmark such as
  HumanEval or a set of real tasks, as upstream did.
  - The device can't see the host cache, so experts that a cheap memcpy could serve are skipped too. A route kernel
    that knows about the host level would lose less quality for the same speed.

### 6.2 Cache-aware substitution (Skliar et al. 2024): rejected
- **Implementation:** a disk-only pick at rank >= R is swapped for the best of the next M runner-ups that is held in
  VRAM or RAM, and the weights are renormalised.
- **Result:** at equal settings it was **slower and had worse KL than plain skipping**. Rank 4: 100 ms at KL 0.025,
  against skipping's 91 ms at KL 0.019.
- **Lesson:** on this model a different expert's output is as wrong as no output, and it costs compute and cache
  churn.

---

## 6b. Second round (2026-10-12): usage-level ideas on top of the best state

The goal of this round was to find gains that come from how the engine is *used*, rather than from the per-token
cycle in isolation. The workload is single-user (one request at a time), so batching was out of scope.

### 6b.1 First, re-profile the best state (split MoE + learned host cache v2 + tail skip 4)
About 92 ms per token under the profiler:

| Stage | ms/token | Share |
|---|---|---|
| PCIe pulls of RAM-tier experts | 27.7 | 30% |
| SSD wait | 19.6 | 21% |
| PCIe pulls of the second pass (experts just read from disk) | 13.1 | 14% |
| GPU compute | ~25 | 27% |

**The bottleneck moved back to the PCIe link:** about 106 experts x 9 MB per token. The tier diagnostics showed where
the remaining SSD reads come from:
- 45%: experts never held in any tier this session;
- 43%: experts evicted from RAM earlier;
- 8%: experts dropped from VRAM.

### 6b.2 A second workload: long coding context
- `bench.py --ctx` uses a frozen 20 KB source file (`prompt_ctx_strata_tokenizer.py.txt`, about 6k tokens) plus a
  code question.
- Its decode speed (11.4 tok/s) matches the short prompt's (11.6). Prefill does not slow the following decode.
- Any idea that depends on the prompt has to be tested on this workload. The short prompt goes through the
  token-by-token path and never reaches prefill.

### 6b.3 Learned host cache v3 (#5): rejected
- **Diagnosis first.** I added counters that classify every RAM-tier eviction as *kept by the host level* or *lost
  to the SSD*. Under v2, 65% were lost.
- **v3 changes:**
  - RAM evicts the coldest *host-shadowed* expert first.
  - The spill thread shadows twice as much per round and ignores the recency window.
- **The mechanism worked:** losses fell from about 5,850 to 718 (8%).
- **The speed didn't:** 11.35 / 11.22 tok/s against v2's 11.58 / 11.68, and 11.07 against 11.39 on the long prompt.
  Host hits rose only ~20%.
- **Why:**
  - The spill now rewrites about 120 host slots per token, so the 1,891-slot level turns over every ~15 tokens. The
    experts it saves are re-used further apart than that.
  - Evicting "shadowed first" sometimes pushes out a hotter RAM expert than LFU would.
  - The doubled memcpy traffic competes with the CPU lane for memory bandwidth.
- **Lesson:** a simulator that predicts a big win from a hierarchy change can still be wrong about the real engine.
  Here it didn't model tail skip, which already removes most of the reads the host level could rescue, nor the
  memory-bandwidth cost of moving data between levels. Validate with the engine's own counters before and after.

### 6b.4 Prompt-informed LFU seeding (#2): rejected, but it found a real flaw in how prefill feeds the LFU
**Offline test first** (`prompt_predict.py`, using the new `STRATA_GLM_PROMPT_COUNTS` dump plus the decode route log
of the same long-prompt run):

| Per-layer budget K | Answer's expert uses covered by the prompt's top-K | By the usage profile's top-K |
|---|---|---|
| 36 (VRAM) | 26.8% | 18.9% |
| 115 (VRAM + pinned RAM) | 61.2% | 51.3% |

- The prompt is a clearly better static predictor than the profile.
- The live LFU still reaches 60% VRAM hit, far above any static set, because it follows locality *within* the answer.
- **75% of the answer's SSD reads were re-reads**, and the prompt's top 150 per layer covered 60% of them, top 200
  covered 78%.

**Root cause found while checking why prefill's counts didn't protect those experts:**
- Prefill adds raw routing counts into the LFU (`glm_prefill.cu`), and the aging rule halves *every* count per
  32,768 events.
- A ~6k-token prompt is about 48,000 events per layer, so roughly 60 halvings over 41 layers.
- Decoding therefore starts with an essentially empty LFU. The prompt's counts **and the usage profile's seeding**
  are both wiped.

**Fix tried** (`STRATA_GLM_PROMPT_LFU=<scale>`): add each layer's prompt routing as a scaled count (0..scale), and
don't age the table during prefill.

**Result:**
- Scale 32: 10.96 / 10.55 tok/s, and the VRAM hit rate fell from ~61% to ~58%.
- Scale 64: 11.11 tok/s.
- Baseline: 10.80 / 11.35.
- So: no gain at either scale, and the cache does slightly worse.

**Lesson:**
- A predictor that beats the old static profile isn't automatically better than an empty, adaptive cache.
- Making prompt-hot experts sticky crowds out the experts the answer actually uses, and the wipe that looked like a
  bug turned out to be harmless here.
- Check the counterfactual against the live adaptive policy, not just against the static warm-up order.

### 6b.5 Ideas from this round that remain untested
- **Idle-time re-tiering** between requests and during tool-call gaps. It must be preemptible: stop issuing moves
  when a request arrives, and stand down before prefill.
- **Per-workload usage profiles**, e.g. separate profiles for coding and chat.
- **A route kernel that knows about the host cache,** so tail skip drops only experts that really hit the SSD.

## 6c. Basic CPU vision on the HIP build (2026-10-12)
The HIP installer turns images off. What was needed:
1. **The encoder for the CPU:** `tools/vision/build_vision_cpu_windows.bat` builds `build-vision/bin/strata-vision.exe`
   (llama.cpp's mtmd plus Maya's `glm5next` patch, built with MSVC).
2. **The mmproj.** Unsloth's `mmproj-model-bf16.gguf` (projector `glm5v`) holds the same tensors, with the same
   names, as Maya's converter output. `tools/vision/mmproj_from_llamacpp.py` rewrites:
   - the projector type to `glm5next`;
   - `swiglu_clamp` to `ffn_clamp` (required: without it the encoder silently drops the SwiGLU clamp);
   - pixel limits to token limits (28x28 pixels per token);
   - BF16 matrices to F16 (the largest |w| is 4.0, well within F16's range).
3. **The vocab GGUF.** The pinned llama.cpp has no `glm5-next` architecture, so
   `tools/vision/vision_vocab_gguf.py` copies the model's own `tokenizer.*` keys into a `glm4` header with
   `n_embd 4096`. `strata-vision` reports that width, and it must match the engine.
4. **Config:** a `"vision"` entry with `"gpu": false`, `"threads": 12`, `"max_tokens": 1024`, plus `--vision` in the
   engine args.
   - Without the token cap, the CPU encoder's warm-up at the largest size (5,329 tokens) took 76 s.

Measured: a 448x448 picture encodes to 256 tokens in 1.7 s on the CPU. End to end, the server described the test
picture correctly in 15 s.

## 7. A reusable method checklist

1. **Read the code paths and the repo's own "tried" notes** first. Write down the per-token data flow and which
   resource each stage uses.
2. **Build a harness:**
   - fixed prompt, fresh process per run, frozen learned state;
   - pinned auto-sized resources;
   - one binary with env-gated features;
   - teacher-forced KL against a reference dump;
   - a functional test.
3. **Measure the noise floor** for both speed (repeat runs) and quality (baseline against itself).
4. **Profile, then convert stage times into bandwidths.** That tells you which hardware limit you are actually at.
5. **Look for saturated resources that run one after another,** and overlap them. Split MoE was the biggest exact
   win.
6. **Check whether the cache keeps learning.** Any bypass that stops admitting misses will collapse the hit rate.
7. **Record a route trace, then simulate policies offline** before writing engine code. Validate the simulator
   against measured counters first.
8. **When the simulator and the engine disagree, add diagnostics.** In this campaign the gap was VRAM drops
   bypassing RAM, found with the tier diagnostics.
9. **For approximations, sweep a knob, plot speed against KL, and pick the knee.** Confirm with a functional task.
   Prefer rules that act only on the slow tier.
10. **After each kept change, re-profile.** The bottleneck moves, and it changes which ideas are worth trying next.

## 8. Most promising next steps
1. **Idle-time re-tiering** between requests and during tool gaps, preemptible (section 6b.5). Untested.
2. **A host-cache-aware route kernel.** Expose a "host" bit so the tail skip only drops experts that would really hit
   the SSD.
3. **Validate tail skip on a broader coding set,** and consider shipping rank 4-6 as an opt-in "fast" mode.
4. **Do not retry without new evidence:** learned host cache v3, and prompt-informed LFU seeding (section 6b).
   The unified-LFU simulation's promise did not survive tail skip and memory-bandwidth effects.
5. **Idle-only, cancellable SSD lookahead.** Low ceiling given the routing analysis; try it only after the above.

## 9. State of the repo after this campaign
- **`main` was not modified by commits.** The user's uncommitted changes were committed onto `exp/baseline`. To
  restore the original working tree:
  ```
  git checkout main
  git checkout exp/baseline -- include src serve tools
  git reset
  ```
- **GitHub:** the user's fork is `CodeXTF2/project-maya`, local remote `mine`; `origin` is the upstream
  `mw00/project-maya`, which isn't the user's. Branch `best-possible-state` (pushed) holds split MoE, learned host
  cache, tail skip and the CPU vision tools. Round 2's v3 and prompt-LFU code is on the local branch
  `exp/hl3-promptwarm`, env-gated.
- **`Start-Maya-Server.bat`** (untracked) sets `STRATA_GLM_SPLIT_MOE=1`, `STRATA_GLM_HOST_LEARN=2`,
  `STRATA_GLM_TAIL_SKIP=4` and `STRATA_GLM_POOL_GB=14`.
- **The memory note** is at `~/.claude/projects/C--Users-Admin-Desktop-AI-project-maya/memory/maya-perf-experiments.md`.
