# VLA Landscape Reference

> Research reference compiled 2026-06-23. Companion to [../README.md](../README.md) and the
> masterplan ([superpowers/plans/2026-06-23-vla-edge-masterplan.md](superpowers/plans/2026-06-23-vla-edge-masterplan.md)).
> Covers: what "language-conditioned" means, simulation environments for
> benchmarking, the benchmark suites themselves, current leaderboard leaders, and
> breakdowns of two key survey papers.

---

## What does "language-conditioned" mean?

A **language-conditioned** policy takes a **natural-language instruction as an
input** and its behavior *depends on* that instruction. The same model produces
different actions for `"pick up the red cube"` vs `"open the drawer"`. The
instruction is the "L" in **V**ision-**L**anguage-**A**ction.

Contrast with policies that are **not** language-conditioned:
- **Single-task** policies — trained to do one thing; no instruction input.
- **Goal-image-conditioned** — told the goal via a target image, not words.
- **One-hot / task-ID** — pick a task from a fixed list by index, not free text.

When a **benchmark** is called "language-conditioned," it means tasks are
specified via language and the eval tests whether the policy actually *follows
the instruction*. Strength varies:
- **Weak**: one fixed instruction string per task (most of LIBERO).
- **Strong**: many paraphrases / compositional, multi-step instructions
  (CALVIN chains, VLABench reasoning tasks).

This matters for us because the demo's "give it a command, it does the thing"
behavior *is* language conditioning — and the realism of that depends on which
benchmark's instruction style we adopt.

---

## Simulation environments (ordered by fit for our goal)

Goal context: deploy & compare small VLAs (e.g. SmolVLA), measure success +
latency, modest/Windows-ish hardware, minimal harness-building.

| # | Sim software | Engine | Lang-conditioned? | Ready VLA integration | Why this rank for us |
|---|---|---|---|---|---|
| **1** | **LIBERO** | robosuite / MuJoCo | ✅ | ✅ OpenVLA, SmolVLA, π0 all report here | **The default.** Lightweight, pip-installable, the standard SmolVLA/OpenVLA are *already* evaluated on — least plumbing, instant apples-to-apples. Downside: nearly saturated (fine for a demo). |
| **2** | **SimplerEnv** | ManiSkill2 / SAPIEN | ✅ | ✅ purpose-built for OpenVLA/RT-1/Octo | Best for evaluating **pretrained** VLAs realistically (sim-real correlation). Use to check "does the off-the-shelf model actually work" before compressing. |
| **3** | **LeRobot sim envs** (`gym-aloha`, `pusht`, `xarm`) | MuJoCo / PyBullet | partial | ✅ **native to SmolVLA** | **Lowest-friction if you stay in the SmolVLA/LeRobot stack** — same repo, same data format, eval scripts included. Narrower task variety / weaker language conditioning. |
| **4** | **ManiSkill 3** | SAPIEN | ✅ | partial (growing) | Modern, fast, **GPU-parallel** rollouts, actively maintained, great rendering. Best if you want throughput/scale; slightly more harness work than LIBERO. |
| **5** | **CALVIN** | PyBullet | ✅ (long-horizon chains) | partial | Long-horizon language sequencing; PyBullet is light and cross-platform. Older ecosystem. |
| **6** | **RoboCasa / RoboCasa365** | robosuite / MuJoCo | ✅ | partial (GR00T-oriented) | Large, realistic kitchen scenes; harder + heavier. Good "impressive demo" visuals. |
| **7** | **Isaac Lab / Isaac Sim** | PhysX | ✅ (DIY) | ❌ build it yourself | Keep only for the **Jetson/NVIDIA deployment story** or GPU-parallel RL. No ready VLA task suite → most setup. |
| **8** | **RLBench** | CoppeliaSim / PyRep | ✅ | ❌ (3D-policy oriented) | 100 tasks but Linux-leaning, painful install; favored by point-cloud policies, not our use case. |
| **9** | **VLABench / Meta-World** | MuJoCo | ✅ | partial | Specialized probes — VLABench for long-horizon *reasoning*, Meta-World for classic multi-task. Use later for a harder test. |

**Pick:** start with **LIBERO**; keep **SimplerEnv** as the realism check and
**LeRobot envs** as the zero-friction fallback.

**Platform caveat:** all are **Linux-first**. MuJoCo/PyBullet run natively on
Windows; robosuite/SAPIEN/CoppeliaSim are smoother on Linux. Run the sim+eval on
the **RTX 4090 PC under Linux or WSL2** and keep heavy work off the 6 GB laptop.

---

## Benchmark suites (the eval targets)

Reminder: a VLA benchmark scores **closed-loop task success** (rollouts in a sim),
not offline dataset accuracy.

| Benchmark | Sim backend | Robot(s) | What it tests | Status 2026 |
|---|---|---|---|---|
| **LIBERO** | robosuite/MuJoCo | Franka | Lifelong manip; 4 suites (Spatial/Object/Goal/Long-10/100) | **Saturated (~96–98%)**; de facto standard |
| **SimplerEnv** | ManiSkill2/SAPIEN | Google Robot, WidowX | Sim eval correlated to real-robot success | Most-used, but noisy (40–99% by setup) |
| **CALVIN** | PyBullet | Franka | Long-horizon language chains (ABC→D) | Near ceiling (~4–4.5 on ABC) |
| **RLBench** | CoppeliaSim | Franka | 100 tasks; favored by 3D policies | Active, niche |
| **ManiSkill 2/3** | SAPIEN | Various | Contact-rich; training + eval | Active, modern |
| **RoboCasa / 365** | robosuite/MuJoCo | Franka + kitchens | Large realistic scenes; 365 tasks | Emerging harder suite |
| **RoboArena** | various | various | Zero-shot generalization focus | Emerging |
| **VLABench** | MuJoCo + dm_control | manip | 100 task cats (60 primitive + 40 composite reasoning), 2000+ objects | Newer (Fudan, Dec 2024) |
| **Meta-World** | MuJoCo | Sawyer | 50 multi-task | Classic, still cited |

---

## Leaderboard leaders (as of mid-2026)

Snapshot — leaderboards update monthly; verify before quoting.

**LIBERO** (basically solved, differences are in the noise):
- **VITA-VLA** — ~97.3% average across all suites (via action-expert distillation).
- **OpenVLA-OFT** and **π0.5** — top-tier reference models.
- **Fast-dVLA** — ~96.6% (discrete-diffusion VLA accelerated to real-time).

**SimplerEnv**:
- **VLA-JEPA** — best on the Google Robot average (wins 2/4 tasks on both Google Robot and WidowX).
- π0-style models report ~65–70%; OpenVLA ~45%; RT-2 ~57% in comparable contexts.

**VLA-Arena** (170 tasks; dimensions: Safety, Distractor, Extrapolation, Long-Horizon; difficulty L0–L2; metrics SR + Cumulative Cost):
- No single dominator — leaders vary by dimension. In one snapshot **Octo-base-1.5** topped total score (~58.3), **OpenVLA-7B-OFT** led orientation accuracy, **π0-fast-droid** led gripper accuracy. Treat as a robustness/safety stress test rather than a single-number ranking.

**Aggregators / harnesses worth using directly:**
- **AllenAI `vla-evaluation-harness`** — "evaluate any VLA on any robot sim benchmark"; a rebuilt leaderboard (~1,885 models × 18 benchmarks, monthly, schema-validated). **Most useful practical tool for our Level 0 rig.**
- **vla-eval** — unified harness, up to **47× faster** LIBERO eval (14 h → 18 min across 6 codebases / 3 benchmarks).
- **tektonian VLA benchmark** — meta-leaderboard aggregating 657 results / 17 benchmarks / 1,704 papers.

> **Takeaway for us:** don't build the eval rig from scratch — start from
> **AllenAI's vla-evaluation-harness** or **vla-eval**. They already wire models
> to sims with success detectors and instrumentation.

---

## Survey breakdown 1 — "VLA in Robotic Manipulation: A Systematic Review" (arXiv 2507.10672)

**Scope:** synthesizes **102 VLA models, 26 datasets, 12 simulation platforms.**

**How it categorizes models.** It does *not* invent named "paradigms"; it organizes
models by **component composition** (its Fig. 8 taxonomy), plus one binary split:

| Axis | Categories | Examples |
|---|---|---|
| Vision encoder | CLIP/SigLIP ViTs · CNNs · DINOv2 | dual SigLIP+DINOv2 (OpenVLA) |
| Language backbone | LLaMA/Vicuna · T5 · Qwen · GPT-based | Llama2 (OpenVLA) |
| Action decoder | Diffusion Transformer · Autoregressive Transformer · MLP | flow/diffusion (π0), discrete tokens (OpenVLA) |
| Build strategy | **End-to-End** (vision+language → action directly) vs **Component-Focused** (improve one building block) | — |

**Two-dimensional dataset characterization.** Each dataset is plotted on:
- **X — Task Complexity (𝒞_task):** breadth/difficulty of the manipulation tasks.
- **Y — Modality Richness (𝒞_mod):** coverage of RGB, depth, proprioception, language.

Its headline gap: **almost no datasets sit in the high-𝒞_task × high-𝒞_mod corner**
(diverse, hard tasks *with* rich multimodal annotation).

### Simulators assessed — complete Table 3 (all 12 platforms)

Columns mirror the survey: sensor modalities, primary use cases, key capabilities,
representative datasets. Cells condensed; nothing dropped.

| Simulator | Modalities | Primary use cases | Key capabilities | Representative datasets |
|---|---|---|---|---|
| **AI2-THOR** | RGB, depth, semantic/instance seg, object states | Embodied nav, object manipulation | Photorealistic indoor scenes; procedural generation; interaction APIs; language/task integration | ALFRED, TEACh, DialFRED |
| **Habitat** | RGB, depth, semantic seg, agent pose | VLN, embodied QA, point-goal nav | Photorealistic high-perf rendering; large-scale 3D scenes; modular sensor/agent APIs | R2R, CVDN, EmbodiedQA |
| **NVIDIA Isaac Sim** | RGB, depth, LiDAR, seg, bbox, point clouds, physics states, force/torque, event camera | RL & control, sim-to-real, synthetic data gen, multi-robot, digital twin, industrial | PhysX dynamics; RTX photorealistic rendering; domain randomization; ROS/ROS2 + Python API; cloud | Open X-Embodiment, Isaac Gym, RLBench, custom |
| **Gazebo** | RGB, depth, LiDAR, IMU, joint states, force/torque, contact, GPS | Control-algo dev, multi-robot coordination, sim-to-real, nav & manip | Open-source; plugin extensibility; multi-physics engines; ROS1/2; multi-robot/sensor | RoboSpatial |
| **PyBullet** | RGB, depth, contact forces, joint states | RL, manip prototyping, physics sim | Real-time physics; Python API; cross-platform; robotics + VR | QUAR-VLA, custom RL/manip |
| **CoppeliaSim** | RGB, depth, joint states, force/torque, proximity | Multi-robot coordination, task scripting, manip, education | Multiple built-in physics engines; remote APIs (Python/ROS/C++); scene editor | RLBench, CALVIN |
| **Webots** | RGB, depth, sound, GPS, proximity, IMU, lidar, joint states | Mobile nav, swarm robotics, manip, education | Cross-platform; rich sensor/actuator models; GUI world design; ROS | AgiBot World |
| **Unity ML-Agents** | RGB, depth, raycasts, physics states | RL & imitation, interactive tasks | Unity visual fidelity; Python/C# APIs; curriculum learning | custom RL/nav (Obstacle Tower, RoomNav, MiniWorld) |
| **MuJoCo** | joint positions, contact forces, kinematics, RGB | Continuous control, dynamics learning, RL | High-speed sim; accurate contact & soft-body; analytic gradients | Meta-World, RoboSuite, custom RL |
| **iGibson** | RGB, depth, semantic & instance masks, object poses, contact forces | Interactive nav, manip, semantic reasoning | Photorealistic dynamic scenes; real-world reconstructions; interactive objects | iGibson v1/v2 |
| **UniSim** | RGB, depth, proprioception, haptics, audio | Multi-modal data gen, multi-agent, manip, nav | Unified multi-sensor API; scalable cloud-native; plugin extensibility | UniSim-VLA |
| **SAPIEN** | RGB, depth, seg masks, contact forces, articulated-object states | Deformable/articulated manip, semantic reasoning, dexterous grasping | High-fidelity GPU physics; real-time; Python API; large articulated-object library | DexGraspNet, TLA |

### Datasets assessed — complete Table 2 (all 26)

Columns mirror the survey: release size, distinctive characteristics, storage
format. (`sim`/`real`/`mixed` inferred from the characteristics column.)

| Dataset (year) | Size | Distinctive characteristics | Format | Sim/Real |
|---|---|---|---|---|
| EmbodiedQA (2018) | 5,000+ QA episodes, 750+ 3D scenes | Goal-directed visual QA in House3D | JSON + egocentric RGB + trajectories | sim |
| R2R (2018) | 7,189 paths, 21,567 instructions | Real-world VLN on Matterport3D, crowd-sourced | JSON + panoramic JPEG + viewpoint graph | real |
| ALFRED (2020) | 8,055 demos, 25,743 directives | Language-conditioned household manip in AI2-THOR, 120 scenes | RGB + interaction masks + JSON actions | sim |
| RLBench (2020) | 100 manipulation tasks | Few-shot/imitation benchmark in PyRep | Pickled demos: joints, images, task desc, proprio | sim |
| CVDN/NDH (2019) | 7,415 instances, 2,050 dialogs | Vision-and-Dialog Navigation from dialog history | JSON dialog turns + paths + image features | real |
| TEACh (2021) | 3,047 two-agent sessions | Multiturn dialog-driven household tasks in AI2-THOR | JSON transcripts + RGB + masks + CSV splits | sim |
| DialFRED (2022) | 53K+ QA pairs, 34K+ tasks | Dialogue-enabled instruction following on ALFRED | dialogue.json + action traces + subgoals | sim |
| Ego4D (2022) | 3,670 h first-person video | Large-scale real egocentric, diverse scenarios | MP4 + JSON narrations + HDF5/LMDB | real |
| CALVIN (2022) | 5,000+ demos | Long-horizon language-conditioned manip | HDF5: RGB-D, proprio, actions, language | sim |
| DROID (2024) | 76k demos; 564 scenes, 86 tasks | High-diversity language-conditioned manip | RLDS: RGB-D, stereo, calib, language, state/action | real |
| Open X-Embodiment (2025) | 1M+ trajectories, 500+ skills, 22 robots | Large-scale multi-embodiment, multi-skill | TFRecords + RLDS: RGB/depth, language, actions | mixed |
| RoboSpatial (2025) | 1M images, 5K 3D scans, 3M spatial QA | 2D-3D paired spatial reasoning | RGB + 3D scans + relational graphs | mixed |
| CoVLA (2024) | 83.3 h driving video, 6M frames | Time-aligned VLA for autonomous driving | RGB video + GPS/IMU trajectories + captions | real |
| TLA (2025) | 30K peg-in-hole demos | Tactile-language-action for insertion/assembly | ROS bags: camera, tactile, lang.json, trajectory | real |
| BridgeData V2 (2023) | 60,096 trajectories (50,365 teleop; 9,731 scripted) | Multi-skill goal- & language-conditioned manip | TFRecords: RGB, language, 7-DoF actions | real |
| LIBERO (2023) | 130 tasks (10 spatial/10 object/10 goal/100 lifelong) | Lifelong VLA benchmark, procedural + declarative | JSON/Parquet: RGB, language, trajectories | sim |
| Kaiwu (2025) | 1M multimodal episodes | Real-world multi-embodiment dexterous manip | HDF5: RGB, depth, 3D skeletons, tactile, EMG, gaze, IMU, audio, language, mocap | real |
| PLAICraft (2025) | 10,000+ h multiplayer Minecraft, 5 modalities | Open-ended multiplayer, emergent tasks, voice-aligned | JSON streams: RGB, audio, keyboard, mouse | sim |
| AgiBot World (2025) | 1M+ dual-arm trajectories | Open-source long-horizon generalist policy learning | ROS: RGB-D, fisheye, tactile, proprio, language | real |
| Robo360 (2023) | 2K+ trajectories, 86 views, 100+ objects | Multimodal for dynamic NeRF, imitation, control | Synced RGB video, depth, audio, proprio, control | real |
| REASSEMBLE (2025) | 4,551 demos, 17 objects, 781 min | Contact-rich multimodal (RGB, audio, event, F/T, proprio) | Synced multistream incl. event camera, mics, F/T (NIST boards) | real |
| RoboCerebra (2025) | 100K long-horizon trajectories, 1K+ tasks | System-2 reasoning & generalization at real-world scale | Plan logs + visual transitions + failure/subtask labels | real |
| IRef-VLA (2025) | 11.5K rooms, 7.6M relations, 4.7M instructions | Imperfect referential grounding in 3D indoor scenes | Scene graphs + free-space maps + affordances | sim |
| Interleave-VLA (2025) | 210K episodes (13M frames) | Interleaved vision-language instruction execution | Images + sketch overlays + text + actions | mixed |
| RoboMM (2024) | 30K sim episodes + 5K real trials | Multimodal fusion: vision, language, proprio, touch | HDF5: rgb, depth, tactile.csv, instructions.json, actions | mixed |
| ARIO (2024) | 50K sim episodes + 5K real trials | Contact-rich manip with tactile, audio, proprio | HDF5: rgb, depth, tactile.csv, instructions.json, actions | mixed |

**Dataset 2-D map (𝒞_task × 𝒞_mod):** the survey plots all of the above on task
complexity × modality richness, flagging the **high×high corner as nearly empty** —
few datasets pair very diverse/hard tasks with rich multimodal annotation. Recurring
problems it names: task diversity, modality imbalance, annotation quality/cost,
realism vs scale.

**Future direction — modular architecture design.** It argues for **plug-and-play
modularity**: vision encoder, language backbone, and action decoder as
**interconnected, independently-upgradable modules** — "swap in a stronger ViT, a
larger LM, or a more expressive diffusion sampler" without full retraining, enabling
fast iteration and cross-embodiment transfer. This directly validates our
`VLAPolicy`-seam plan (and the Level 2 Branch 2 backbone swaps).

---

## Survey breakdown 2 — "An Anatomy of VLA Models: From Modules to Milestones and Challenges" (arXiv 2512.11362)

**Structure:** organized as a researcher's learning path — **Modules → Milestones
→ Challenges**, framed around the **perception–action loop**; live website.

Five challenges total — **Representation, Execution**, Generalization, Safety,
Dataset & Evaluation. The two most relevant to us, elaborated:

**Representation (§4.1) — multi-modal alignment & physical-world modeling.** Five sub-problems:
- **Vision–language gap:** make visual features responsive to the instruction; use the LM as an intermediate symbolic representation over observations.
- **Vision–language–action gap:** ground semantics into motor commands — either end-to-end (discretize actions as tokens the VLM emits) or via explicit hierarchical planning layers between language and control.
- **Multi-modal sensory fusion:** go beyond RGB (tactile, force, audio for contact-rich tasks); per-modality encoders aligned by contrastive learning; fusion from deep integration to modular mixture-of-experts.
- **Spatial-temporal representation:** 2D → 3D (depth, point clouds, voxels, occupancy); predict 4D point-motion over time; integrate via 3D adapters, multi-view, or reprojecting 3D into 2D.
- **Dynamic world modeling:** predict future states (pixel or latent) as an auxiliary signal; explicit internal rollouts to evaluate candidate actions.

**Execution (§4.2) — instruction following, planning, robust real-time control.** Four sub-problems:
- **Parsing complex instructions:** multimodal prompts (text + images + sketches), ambiguous commands ("clean this"), active clarification.
- **Hierarchical planning / task decomposition:** long-horizon → subgoals via linguistic sub-steps, pixel-level subgoal images / affordance chains, or compositional skill libraries.
- **Error detection & autonomous recovery:** detect anomalies mid-task; human-in-the-loop correction; self-correction without intervention.
- **Real-time execution & compute efficiency** — *directly our project*: **static** (quantization, compression, lightweight backbones, **linear attention**), **dynamic** (layer skipping, early exit, token pruning/caching, accelerated decoding), **action-representation** (efficient tokenization, asynchronous execution), **training-paradigm** (reasoning traces at train time, dropped at inference; dual-system architectures).

This last bullet is essentially the menu of techniques for Level 2 (compression + SSM) —
note "linear attention" is where the SSM bet lives.

---

## Sources

- Surveys: [VLA Systematic Review (2507.10672)](https://arxiv.org/abs/2507.10672) · [Anatomy of VLA Models (2512.11362)](https://arxiv.org/abs/2512.11362)
- Leaderboards/harnesses: [VLA-Arena](https://vla-arena.github.io/) · [VLA-Arena paper (2512.22539)](https://www.emergentmind.com/papers/2512.22539) · [AllenAI vla-evaluation-harness](https://github.com/allenai/vla-evaluation-harness) · [vla-eval (2603.13966)](https://arxiv.org/pdf/2603.13966) · [tektonian VLA benchmark](https://tektonian.com/vla-benchmark)
- Models referenced: VITA-VLA (2510.09607), Fast-dVLA (2603.25661), VLA-JEPA (2602.10098), VLANeXt (2602.18532).
- Field state: [State of VLA Research at ICLR 2026 — M. Reuss](https://mbreuss.github.io/blog_post_iclr_26_vla.html)
