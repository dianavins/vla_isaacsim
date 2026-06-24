# Edge VLA Research Masterplan

> **For agentic workers:** This is a **masterplan / research roadmap**, not a single
> step-by-step implementation plan. It decomposes the program into levels, locks the
> decisions made during brainstorming, and defines the shared foundation (KPIs, the
> Jetson constraint model, and reusable subroutines). Each *level* gets its own detailed
> implementation plan written separately (see **Plan Index**). Level 0 is fully
> specifiable now and should be planned/executed first.

**Goal:** Find a Vision-Language-Action (VLA) model that runs near-real-time on edge
(Jetson-class) hardware — anywhere from a ~5 MB tiny model up to a ~7B-param model that
fits under quantization — while retaining as much task capability as possible, by
systematically comparing existing transformer VLAs, compressed variants, and
State-Space-Model (SSM) VLAs on **one common benchmark: LIBERO**.

**Architecture:** Every candidate model — original, compressed, or SSM — runs on **LIBERO**
so all results are apples-to-apples, and is scored by a single shared **KPI-Assessment**
subroutine against a **3-light Jetson constraint model**. Compression is an **evidence-driven
research loop with an explicit exit criterion**, applied as a reusable **Compression-Loop**
subroutine. Once final compressed models that meet our standards are found, two hardware
backends (custom RTL; Akida neuromorphic) sit at the end and consume only the model tracks
that are physically compatible with them.

**Tech Stack:** Python, PyTorch; **LIBERO** (Franka + MuJoCo) for sim; **`vla.cpp` /
`vla-evaluation-harness`** as a unified VLA runtime over LIBERO so heterogeneous
checkpoints (OpenVLA, SmolVLA, SSM models) run behind one interface;
quantization/pruning/distillation tooling (bitsandbytes / GPTQ / AWQ / torch.ao / custom);
single NVIDIA RTX 4090 for all training and desktop measurement. Jetson and Akida are
**modeled/estimated**, not measured.

---

## Global Constraints

- **One benchmark — LIBERO.** Every model that is meant to be *compared* runs on LIBERO,
  using its standard 4 task suites and closed-loop success detectors. There is no second
  embodiment/sim; IsaacSim/IsaacLab is **not** required (the repo name is historical).
- **Reuse the eval harness, don't build it.** Adopt **`vla.cpp` / `vla-evaluation-harness`**
  as the unified VLA runtime + eval layer over LIBERO. We own only the task selection,
  KPI instrumentation, and the Jetson constraint model — not the rollout machinery.
- **Compute budget:** single RTX 4090 for all training, fine-tuning, and desktop KPI
  measurement. Anything requiring multi-GPU pretraining scale is **out of scope**.
- **No edge hardware:** no Jetson board and no Akida board. All Jetson and Akida KPIs are
  **ESTIMATES**, and every estimated number MUST be flagged `[EST]` in results tables.
  (Akida access may become available later *if* the SSM track shows very promising results;
  ignore Akida until then.)
- **Open artifacts only:** a model/architecture is in scope only if it has **open weights
  or an open, reproducible architecture**. Research papers without released
  weights/architecture (e.g. LiteVLA / LiteVLA-Edge unless they publish) are **excluded**
  until/unless artifacts appear. Verify before committing to any model.
- **Single owner, hobby project:** no external review gates, no deadline gates. RTL is a
  planned (not "aspirational/dashed") deliverable.
- **Distillation-only for custom SSM backbones:** training a competitive vision/language
  SSM **from scratch is out of scope**; only distillation from a teacher is allowed.

---

## Level 0 — Shared Foundation (build first, reuse everywhere)

Level 0 is setup **plus** the reusable artifacts every later level depends on. It is the
only level fully specifiable today and gets the first detailed plan.

### 0.1 Environment & sim setup
- Install PyTorch + CUDA on the 4090 host.
- Install **LIBERO** and verify a random/scripted policy runs an episode with success
  detection across all four task suites.
- Install the unified runtime (**`vla.cpp` / `vla-evaluation-harness`**) and confirm a
  reference checkpoint (e.g. SmolVLA) loads, steps, and reports a success rate on LIBERO.
- Pin all versions; record a reproducible environment spec.

### 0.2 KPI glossary (define once, reuse at every assessment node)
Every model, at every stage, is described by exactly this row:

| KPI | Unit | Definition | Source |
|-----|------|-----------|--------|
| `success_rate` | % | per-suite and average task success (fixed trials/task) | measured in sim |
| `control_rate` | Hz | closed-loop action inferences per second | measured (desktop) / `[EST]` (Jetson) |
| `latency_p50/p95` | ms | single action-inference latency | measured (desktop) / `[EST]` (Jetson) |
| `peak_mem` | GB | peak VRAM / unified memory during inference | measured (desktop) / `[EST]` (Jetson) |
| `params` | M / B | total parameter count | static |
| `disk_size@quant` | MB | on-disk size at given quantization | static |
| `energy_per_inference` | J | avg_power × latency | `[EST]` (Jetson) |
| `avg_power` | W | average inference power draw | `[EST]` (Jetson) |

Desktop (4090) numbers are real measurements. **Jetson power and latency are always
`[EST]`** until/unless a board is obtained.

### 0.3 The 3-light Jetson constraint model (estimated)
A "fits ✓/✗" flag hides the real story (a model can fit in memory yet run below the control
floor), so the Jetson gate is **three lights**, not one. It is a pure model (no board).
Target SKU is a **flag/config** (default **Jetson Orin Nano 8GB**; secondary column
**Orin NX 16GB**). For a candidate model it emits three booleans:

- **Light 1 — Fits:** `peak_mem_est ≤ device_mem_budget`
  where `peak_mem_est = weights(@quant) + activations + KV/SSM-state + framework overhead`.
- **Light 2 — Real-time:** `control_rate_est ≥ 10 Hz`
  estimated from model FLOPs ÷ device throughput (TOPS) with a utilization derate, cross-checked against desktop latency scaled by a roofline factor.
- **Light 3 — Power:** `energy_per_inference_est` within budget and `avg_power_est ≤` module envelope (e.g. ≤15 W).

Output verdict per model: `(✓/✗, ✓/✗, ✓/✗)` per target SKU. **A model that fits but runs at
6.6 Hz is a `(✓, ✗, …)` — exactly the failure mode the single-flag version would have
hidden.**

### 0.4 Reusable subroutines (shared stages, not duplicated boxes)
KPI assessment and the compression loop are **reusable subroutines** the levels call into,
not boxes re-drawn per branch:

- **`KPI-Assessment(model, env, device_profile)` → KPI row + 3-light verdict.** Called by
  every baseline and every compressed/SSM candidate.
- **`Compression-Loop(model, target)` → best Pareto point + full trajectory.** An
  **evidence-driven research loop** (skeleton in §0.5; *expect to refine how we do this*).
  **Defined here in §0.5; first built & validated in Level 2 Branch 1 on the transformer
  baselines; reused unchanged in Level 2 Branch 2 on the SSM VLAs (and later as Akida
  prep).** It is foundation, not owned by any one branch — that is what keeps the
  compression logic from being duplicated across the two Level-2 branches.

### 0.5 Compression-Loop skeleton (provisional — to be refined)
The compression loop is the **scientific method applied to compression**: don't run a fixed
menu of tricks blindly — *reason about* which compressions should help, *test them in a
controlled sweep*, and let the **measured evidence shape the next round**. One round `k`:

1. **Design (reason + research).** Propose a candidate strategy set `S_k`. Each candidate is
   justified by **(a) mathematical/architectural reasoning** about *why* it should cut
   size/latency/power while preserving capability (where the params/FLOPs/memory actually go;
   what the quantization error or pruned subspace does to the action distribution), and
   **(b) prior knowledge of compression** — drawn from the published literature (DyQ-VLA,
   AWQ/GPTQ, NF4, LiteVLA, ActionFlow, distillation work…) **and from Claude's own
   internalized understanding of compression techniques; sources need not be peer-reviewed.**
   For `k = 0` this is the first principled set; for `k > 0` it is **shaped by the evidence
   from round `k-1`**.
2. **Sweep (ablation study).** Apply each candidate — and controlled combinations /
   one-factor-at-a-time variants — to the model, running `KPI-Assessment` on **every**
   variant so each knob's effect is **isolated**, not confounded.
3. **Analyze (extract evidence).** Read off which knobs move which KPIs, where capability
   breaks, and how strategies interact — i.e. which moves contribute to the
   size-vs-success Pareto frontier, and *why*.
4. **Synthesize next set.** Combine that evidence with further reasoning + research to design
   `S_{k+1}` (push the knob that paid off, drop the one that didn't, test a hypothesis the
   ablation raised).
5. **Check exit criteria** (see **Exit Criteria** below). If neither fires, `k ← k+1` and
   repeat.

Record **every variant's KPI row**, not just the winner, so the whole trajectory is
auditable evidence (and feeds the next round). This skeleton is deliberately provisional —
refining *how* each step is done (which reasoning, which ablation design, how evidence is
scored) is itself part of the Level 2 Branch 1 work.

---

## The DAG (levels)

```
LEVEL 0  Shared Foundation
   ├─ setup (LIBERO + vla.cpp / vla-evaluation-harness)
   ├─ KPI glossary            ─┐
   ├─ Jetson 3-light model    ─┼─ reused by every assessment node
   └─ KPI-Assessment + Compression-Loop subroutines ┘  (Compression-Loop defined §0.5)

LEVEL 1  Baselines on LIBERO
   ├─ OpenVLA   → KPI-Assessment
   └─ SmolVLA   → KPI-Assessment
      (LiteVLA / -Edge only if open weights exist — verify first)
   Reference target (OpenVLA, published): Spatial 84.7% · Object 88.4% ·
   Goal 79.2% · Long 53.7% · avg 76.5%. These are the fixed numbers every
   later level measures against.

LEVEL 2
   ├─ BRANCH 1  Compression toolchain
   │     Builds + validates the evidence-driven Compression-Loop (§0.5) on the
   │     Level-1 transformer models (OpenVLA, SmolVLA).
   │     ⇒ deliverable = a working, instrumented compression toolchain +
   │       Pareto results — NOT a smaller SmolVLA for its own sake. This is the
   │       method/infra reused on the SSM models and (later) for Akida.
   │
   └─ BRANCH 2  SSM exploration  (all evaluated on LIBERO)
         ├─ 2a  Pre-existing SSM VLAs/VLMs (e.g. RoboMamba) → KPI-Assessment
         ├─ 2b  Assemble VLA from pretrained SSM vision + language backbones ┐
         └─ 2c  Distill small SSM vision/language backbones from a teacher  ┘
                         │ (2b + 2c reconverge)
                         ▼
                 [VLM / vision+language → VLA]   ← FOGGY node, method TBD
                 (action head/tokenizer + action-data fine-tuning)
                         ▼
                   KPI-Assessment on LIBERO
                         ▼
                 rejoin 2a  →  Compression-Loop (§0.5, reused) on all SSM VLAs

LEVEL 3  Hardware backends (parallel; consume best models)
   ├─ BRANCH 1  RTL   : uncover low-level ops → write RTL for best models   [planned]
   └─ BRANCH 2  Akida : map SSM/TENNs-compatible models only  [EST; deferred until promising]
```

### Why Akida only consumes the SSM track
Akida supports specific (TENNs/SSM-style) layer types; a transformer like OpenVLA will not
map. So the Akida backend draws **only** from Branch-2 outputs — never from the
transformer-compression (L2-B1) outputs. RTL has no such restriction.

---

## Exit Criteria (for the Compression-Loop)

The compression loop needs both a *success* exit and a *give-up* exit, or it runs forever:

1. **SUCCESS exit:** stop when the candidate **passes all 3 Jetson lights** for the target
   SKU **AND** retains **≥90% of the baseline average success** (≤10% relative success
   drop). This is the win condition.
2. **DIMINISHING-RETURNS exit:** stop iterating a model when its **size-vs-success Pareto
   frontier improves by <2% over 2 consecutive iterations**. This guarantees termination
   when the success gate is unreachable.

The loop terminates on **whichever fires first**; always record the best Pareto point
reached, even on a give-up. It is a cycle gated by these two conditions, not a node.

---

## Foggy / deferred / out-of-scope

- **`VLM → VLA` node (foggy):** needs an action head/tokenizer + action-data fine-tuning
  (FAST tokenizer, action expert, or diffusion head). Method chosen when Branch 2 reaches
  it; represented as a real node now, not a connector arrow.
- **Akida (deferred):** estimated only; do not invest until the SSM track shows promise.
- **From-scratch SSM backbones (out of scope):** the 4090 can't pretrain competitive
  vision/language SSMs. Branch 2c is **distillation-only**.
- **LiteVLA / LiteVLA-Edge (excluded):** unless open weights/architecture are released.

---

## Plan Index (detailed plans, written per level)

Each entry below becomes its own `docs/superpowers/plans/…` document, produced when its
predecessors are far enough along. Level 0 is ready to plan now.

1. **`…-level0-foundation.md`** — env/sim setup, KPI glossary, Jetson 3-light model,
   `KPI-Assessment` + `Compression-Loop` (§0.5) subroutines. *(write next)*
2. **`…-level1-baselines.md`** — OpenVLA + SmolVLA on LIBERO.
3. **`…-level2-branch1-compression.md`** — build/validate the evidence-driven
   Compression-Loop on the Level-1 models.
4. **`…-level2-branch2-ssm.md`** — 2a pre-existing SSM VLAs, 2b assembly, 2c distillation,
   VLM→VLA node, SSM compression (reuse Compression-Loop).
5. **`…-level3-rtl.md`** and **`…-level3-akida.md`** — hardware backends.

---

## Self-Review (masterplan ↔ brainstorming spec)

- ✅ One benchmark — LIBERO; no second embodiment, IsaacSim not required.
- ✅ RoboCasa/GR00T reference lane removed entirely (extra, and weak absolute numbers).
- ✅ Eval harness reused (`vla.cpp` / `vla-evaluation-harness`), not hand-rolled.
- ✅ 3-light Jetson gate, estimated, with the "fits but too slow" failure mode called out.
- ✅ L2-B1 reframed to "build the compression toolchain," deliverable stated.
- ✅ Compression is an evidence-driven research loop (§0.5) with an explicit dual exit criterion.
- ✅ Compression-Loop defined once (§0.5), built in L2-B1, reused in L2-B2 — not duplicated.
- ✅ Branch 2c restricted to distillation; from-scratch out of scope.
- ✅ VLM→VLA is an explicit (foggy) node, not a connector.
- ✅ RTL planned (not dashed); RTL and Akida are parallel alternatives; Akida only consumes the SSM track.
- ✅ Level 0 = setup + shared foundation; KPIs/subroutines defined once and reused.
- ✅ LiteVLA/-Edge gated on open artifacts.
