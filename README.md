# Edge VLA

> The end goal is a VLA (Vision-Language-Action) model that fits on edge — anywhere from
> ~5 MB up to a ~7B-param model under quantization — running as close to real time as
> possible. *(The repo name `vla_isaacsim` is historical; IsaacSim/IsaacLab is **not** in
> the validation path — see below.)*

**The plan lives in two companion docs:**
- **[docs/superpowers/plans/2026-06-23-vla-edge-masterplan.md](docs/superpowers/plans/2026-06-23-vla-edge-masterplan.md)** — the masterplan: the Level 0–3 DAG, KPI definitions, the 3-light Jetson model, and the Compression-Loop exit criteria.
- **[docs/vla-landscape.md](docs/vla-landscape.md)** — research reference: sim environments, benchmark suites, leaderboards, and survey breakdowns.

This README holds the **project context** the plan assumes but doesn't restate: the
hardware reality, the framing decisions, related work to build on, and the open questions.

---

## End goal (in full)

A **working edge VLA demo** running on **Jetson or Akida**, made **as small and as capable
as possible**. The objective is a deployable demo plus the knowledge of which
compression/architecture techniques actually move the needle — **not a research paper and
not a novel benchmark result.** Because novelty is not a constraint, existing tiny VLAs
(SmolVLA, TinyVLA, LiteVLA) are **starting points to build on**, not competitors to beat.
Validated on **LIBERO**.

## Hardware reality

| Topic | Decision / fact |
|---|---|
| Edge hardware target | **Jetson or Akida** (Jetson is the more tractable first demo target; Akida is the lower-power stretch). Final choice still open and may be both. |
| Jetson board | **No board on hand → all Jetson KPIs are estimated** (`[EST]`). Modeled via the 3-light constraint model in the masterplan. |
| Akida (BrainChip) | **No board on hand → estimated only; access obtainable later *if* the SSM track shows very promising results.** The neuromorphic/SNN toolchain (`akida_env`, `ann_to_snn_testing`, `dvs_test`, `snnTag`) already exists locally; ignore Akida until the SSM track is promising. |
| Dev laptop | **RTX 4050 Laptop GPU, 6 GB VRAM.** torch 2.7.0+cu128 works. |
| Remote GPU | **"Dad's PC" — RTX 4090, 24 GB VRAM.** Can host OpenVLA-7B *unquantized* (~14–16 GB) with headroom; natural **remote policy server** for the OpenVLA backend while the laptop runs LIBERO. All training/measurement targets the 4090. |
| 6 GB ceiling | LIBERO + OpenVLA-7B (~7 GB at 4-bit) **cannot coexist on the laptop.** OpenVLA must run on the remote 4090. |
| Sim environment | **LIBERO** (Franka + MuJoCo), the standard closed-loop VLA suite. Reuse an existing runtime (`vla.cpp` / `vla-evaluation-harness`); we own only task selection + KPI instrumentation. |

## Key framing notes

- **"5 MB to 7 B" is a ~1400× range spanning different deployment classes.** The final edge
  target collapses it. Until decided, design so hardware is swappable.
- **OpenVLA is a teacher / accuracy ceiling, not an edge baseline.** It is Llama2-7B + dual
  vision encoders, ~5–10 Hz on a good GPU, with embodiment-specific discretized action
  tokens. Useful as a distillation teacher.
- **SmolVLA (~450 M)**, TinyVLA, RoboMamba are **starting points to deploy/build on**, not
  baselines to beat. Standing one up is the fastest route to a demo.
- **SSM rationale (Mamba / Mamba-2 / VMamba / Jamba / TENNs):** the headline
  linear-vs-quadratic win matters most for long sequences; for a VLA step the bigger edge
  wins are **constant-memory recurrent inference (no growing KV cache)** and **clean mapping
  onto TENNs-style hardware**. Frame SSM as a deployment/hardware play, not primarily a
  FLOPs play.
- **A VLA benchmark is closed-loop task success, not dataset accuracy.** Load the policy
  into a simulator, give it an instruction + live observations, let it act over many steps,
  and measure **success rate** (+ a progress score), averaged over trials/seeds. Offline
  action-MSE/token-accuracy correlates poorly with real task success — which is why LIBERO
  (closed-loop rollouts) is the benchmark.
- **"Tiny VLA at ~90% real-time" is already done** (SmolVLA, TinyVLA, EdgeVLA, LiteVLA/-Edge,
  DyQ-VLA, ActionFlow). All target the **watts** regime (Jetson/RPi). The under-explored seam
  is the **milliwatt / neuromorphic / event-driven** regime (Akida/TENNs, DVS input,
  SSM/spiking) — relevant if we push toward the Akida demo. Grounding number: **OpenVLA-7B ≈
  3 FPS on Jetson AGX Orin even at INT4**; control wants ≥10 Hz, ideally 20–30 Hz.

## Related work — build on these, don't re-derive

Edge/tiny VLA is a crowded area as of 2026; treat these as starting points and references.

| Work | What it is | Why it matters to us |
|---|---|---|
| **SmolVLA (~0.45B)** | Compact VLA; matches larger models on LIBERO/Meta-World; real-time on CPU; ONNX + quant. | Strongest candidate to deploy first. |
| **TinyVLA / EdgeVLA** | Small, data-efficient VLAs for edge. | Alternative small backbones. |
| **LiteVLA** | CPU-only via NF4 quant + llama.cpp; async control on **Raspberry Pi 4**. | Reference for extreme low-end deployment + async control loop. (Use only if open weights exist.) |
| **LiteVLA-Edge** | Fully on-device on **Jetson Orin**. | Reference target/pipeline for our Jetson demo. (Use only if open weights exist.) |
| **DyQ-VLA** | Kinematics-aware dynamic quantization for real-time edge. | Compression technique to try in Level 2 Branch 1. |
| **ActionFlow** | System-level pipelining to cut autoregressive latency. | Inference-system trick if latency-bound. |
| **RoboMamba** | Mamba/SSM-based VLA. | Reference for the Level 2 Branch 2 SSM track (not edge-optimized). |
| Surveys | "VLA Systematic Review" (2507.10672), "Anatomy of VLA Models" (2512.11362). | Catalog of techniques; read before reinventing. |

Leaderboards/harnesses (deeper detail in [docs/vla-landscape.md](docs/vla-landscape.md)):
**AllenAI `vla-evaluation-harness`**, **vla-eval** (unified harness), **VLA-Arena**,
**tektonian VLA benchmark** — read before standing up the Level 0 rig.

## Open questions / parking lot

- Final demo hardware: Jetson first (which class — Orin Nano / AGX Orin?), Akida as the stretch.
- Whether the first demo even needs OpenVLA, or should start directly from SmolVLA.
- Level 0 architecture: decoupled `VLAPolicy` seam + Dummy backend (recommended) vs. monolithic-first.
- Which SSM backbone(s) to prioritize in Level 2 Branch 2 (Mamba-2 LLM, VMamba vision, RoboMamba end-to-end, Jamba hybrid).
- Distillation strategy details (token-level vs. feature-level; teacher = OpenVLA vs. a compressed model).
- Whether to bring event-camera (DVS) input into the Akida track (Level 3 Branch 2).
