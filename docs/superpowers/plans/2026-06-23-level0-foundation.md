# Level 0 — Shared Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the reusable foundation every later level calls into — the KPI data model, the estimated 3-light Jetson constraint model, decoupled policy/environment seams, the `KPI-Assessment` subroutine, and the evidence-driven `Compression-Loop` skeleton (control flow + Pareto/exit-criteria machinery) — plus a verified LIBERO + `vla-evaluation-harness` install.

**Architecture:** A small Python package `edgevla` under a `src/` layout. **Pure-logic components** (KPI row, model stats, device profiles, the Jetson gate, Pareto frontier, exit criteria, the compression-loop control flow) are fully unit-tested with synthetic numbers, so the math is verified independently of any real Jetson spec or external sim. **Integration components** (LIBERO, the `vla-evaluation-harness` runtime, real desktop latency/memory measurement) sit behind two narrow seams — `VLAPolicy` and `EnvAdapter` — implemented first against `DummyPolicy`/`FakeEnv` (unit-tested) and only then wired to the real sim (smoke-verified by running, not by unit tests). This is the "decoupled `VLAPolicy` seam + Dummy backend" option from the masterplan's open questions.

**Tech Stack:** Python ≥3.10, PyTorch 2.7.0+cu128 (already installed), pytest, NumPy, dataclasses + `typing.Protocol`. LIBERO (Franka + MuJoCo) for sim; `vla.cpp` / `vla-evaluation-harness` as the unified runtime. Single RTX 4090 for measurement.

## Global Constraints

- **One benchmark — LIBERO.** Every comparable model runs on LIBERO; IsaacSim/IsaacLab is **not** required. No second embodiment/sim.
- **Reuse the eval harness, don't build it.** Adopt `vla.cpp` / `vla-evaluation-harness` over LIBERO; we own only task selection, KPI instrumentation, and the Jetson model.
- **Compute budget:** single RTX 4090 for all training/measurement. No multi-GPU pretraining-scale work.
- **No edge hardware:** no Jetson/Akida board. All Jetson KPIs are **ESTIMATES** and must be flagged `[EST]`. The Jetson estimates live in `GateVerdict`; desktop-measured numbers live in `KPIRow` and carry an empty `estimated` set.
- **Open artifacts only:** models need open weights or open, reproducible architecture.
- **Single owner, hobby project:** no external review/deadline gates.
- **Python ≥3.10, `src/` layout, type hints on all public functions, pytest for every pure-logic unit.**
- **Default Jetson SKU = Orin Nano 8GB; secondary = Orin NX 16GB.** Real-time floor = **10 Hz**. Power budget default = **15 W**. Compression-loop: success exit retains **≥90%** of baseline success; diminishing-returns exit at **<2%** Pareto-hypervolume improvement over **2** rounds.

---

## File Structure

```
pyproject.toml                         # package + pytest config
src/edgevla/__init__.py
src/edgevla/kpi.py                     # KPIRow dataclass (the one KPI row)
src/edgevla/modelstats.py              # ModelStats (params/bytes/flops) + peak-mem helper
src/edgevla/devices.py                 # DeviceProfile + estimated Orin Nano/NX profiles
src/edgevla/jetson.py                  # three_light_gate() + GateVerdict
src/edgevla/policies/__init__.py
src/edgevla/policies/base.py           # VLAPolicy Protocol
src/edgevla/policies/dummy.py          # DummyPolicy (plumbing backend)
src/edgevla/envs/__init__.py
src/edgevla/envs/base.py               # EnvAdapter Protocol
src/edgevla/envs/fake.py               # FakeEnv (deterministic, for tests)
src/edgevla/envs/libero.py             # LIBERO adapter (integration; smoke-tested)
src/edgevla/assessment.py              # kpi_assessment() -> (KPIRow, GateVerdict)
src/edgevla/compression/__init__.py
src/edgevla/compression/pareto.py      # ParetoPoint, pareto_frontier(), hypervolume()
src/edgevla/compression/exit_criteria.py  # success_exit(), diminishing_returns_exit()
src/edgevla/compression/loop.py        # compression_loop() skeleton + LoopResult
scripts/smoke_libero.py                # end-to-end smoke run (DummyPolicy on LIBERO)
tests/test_kpi.py
tests/test_modelstats.py
tests/test_devices.py
tests/test_jetson.py
tests/test_policies.py
tests/test_envs_fake.py
tests/test_assessment.py
tests/test_pareto.py
tests/test_exit_criteria.py
tests/test_loop.py
```

---

### Task 1: Project scaffolding & test harness

**Files:**
- Create: `pyproject.toml`
- Create: `src/edgevla/__init__.py`
- Create: `tests/test_smoke.py`

**Interfaces:**
- Consumes: nothing.
- Produces: an importable `edgevla` package (`edgevla.__version__: str`) and a working `pytest` run.

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "edgevla"
version = "0.0.1"
requires-python = ">=3.10"
dependencies = ["numpy>=1.24"]

[project.optional-dependencies]
dev = ["pytest>=8"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

- [ ] **Step 2: Create the package init**

`src/edgevla/__init__.py`:

```python
__version__ = "0.0.1"
```

- [ ] **Step 3: Write the smoke test**

`tests/test_smoke.py`:

```python
import edgevla


def test_package_imports():
    assert edgevla.__version__ == "0.0.1"
```

- [ ] **Step 4: Install dev deps and run the test**

Run:
```bash
python -m pip install -e ".[dev]"
python -m pytest tests/test_smoke.py -v
```
Expected: PASS (1 passed). If `pytest` import of `edgevla` fails, confirm `pythonpath = ["src"]` is present.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/edgevla/__init__.py tests/test_smoke.py
git commit -m "chore: scaffold edgevla package + pytest"
```

---

### Task 2: `KPIRow` — the one KPI row

**Files:**
- Create: `src/edgevla/kpi.py`
- Test: `tests/test_kpi.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `KPIRow` dataclass with fields `model_name: str`, `device_name: str`, `success_avg: float`, `success_per_suite: dict[str, float]`, `control_rate_hz: float`, `latency_ms_p50: float`, `latency_ms_p95: float`, `peak_mem_gb: float`, `params_m: float`, `disk_size_mb: float`, `energy_j: float | None`, `avg_power_w: float | None`, `estimated: frozenset[str]`; method `to_dict() -> dict`.

- [ ] **Step 1: Write the failing test**

`tests/test_kpi.py`:

```python
from edgevla.kpi import KPIRow


def make_row(**over):
    base = dict(
        model_name="smolvla",
        device_name="rtx4090",
        success_avg=0.76,
        success_per_suite={"spatial": 0.84, "object": 0.88},
        control_rate_hz=12.0,
        latency_ms_p50=80.0,
        latency_ms_p95=110.0,
        peak_mem_gb=2.1,
        params_m=450.0,
        disk_size_mb=900.0,
    )
    base.update(over)
    return KPIRow(**base)


def test_defaults_have_no_estimated_fields():
    row = make_row()
    assert row.energy_j is None
    assert row.avg_power_w is None
    assert row.estimated == frozenset()


def test_to_dict_roundtrips_core_fields():
    row = make_row(energy_j=0.5, avg_power_w=15.0, estimated=frozenset({"energy_j"}))
    d = row.to_dict()
    assert d["model_name"] == "smolvla"
    assert d["success_avg"] == 0.76
    assert d["estimated"] == ["energy_j"]
    assert d["success_per_suite"]["spatial"] == 0.84
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_kpi.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'edgevla.kpi'`.

- [ ] **Step 3: Write minimal implementation**

`src/edgevla/kpi.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class KPIRow:
    model_name: str
    device_name: str
    success_avg: float
    success_per_suite: dict[str, float]
    control_rate_hz: float
    latency_ms_p50: float
    latency_ms_p95: float
    peak_mem_gb: float
    params_m: float
    disk_size_mb: float
    energy_j: float | None = None
    avg_power_w: float | None = None
    estimated: frozenset[str] = field(default_factory=frozenset)

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "device_name": self.device_name,
            "success_avg": self.success_avg,
            "success_per_suite": dict(self.success_per_suite),
            "control_rate_hz": self.control_rate_hz,
            "latency_ms_p50": self.latency_ms_p50,
            "latency_ms_p95": self.latency_ms_p95,
            "peak_mem_gb": self.peak_mem_gb,
            "params_m": self.params_m,
            "disk_size_mb": self.disk_size_mb,
            "energy_j": self.energy_j,
            "avg_power_w": self.avg_power_w,
            "estimated": sorted(self.estimated),
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_kpi.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/edgevla/kpi.py tests/test_kpi.py
git commit -m "feat: KPIRow data model"
```

---

### Task 3: `ModelStats` — params / bytes / flops accounting

**Files:**
- Create: `src/edgevla/modelstats.py`
- Test: `tests/test_modelstats.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `ModelStats` dataclass with fields `params: int`, `weight_bytes: int`, `activation_bytes: int`, `state_bytes: int`, `flops_per_step: int`; method `peak_mem_bytes(overhead_bytes: int = 0) -> int`.

- [ ] **Step 1: Write the failing test**

`tests/test_modelstats.py`:

```python
from edgevla.modelstats import ModelStats


def test_peak_mem_sums_components_plus_overhead():
    s = ModelStats(
        params=450_000_000,
        weight_bytes=900_000_000,
        activation_bytes=300_000_000,
        state_bytes=50_000_000,
        flops_per_step=2_000_000_000,
    )
    assert s.peak_mem_bytes() == 1_250_000_000
    assert s.peak_mem_bytes(overhead_bytes=250_000_000) == 1_500_000_000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_modelstats.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'edgevla.modelstats'`.

- [ ] **Step 3: Write minimal implementation**

`src/edgevla/modelstats.py`:

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelStats:
    params: int
    weight_bytes: int
    activation_bytes: int
    state_bytes: int
    flops_per_step: int

    def peak_mem_bytes(self, overhead_bytes: int = 0) -> int:
        return (
            self.weight_bytes
            + self.activation_bytes
            + self.state_bytes
            + overhead_bytes
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_modelstats.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src/edgevla/modelstats.py tests/test_modelstats.py
git commit -m "feat: ModelStats with peak-memory helper"
```

---

### Task 4: `DeviceProfile` + estimated Jetson profiles

**Files:**
- Create: `src/edgevla/devices.py`
- Test: `tests/test_devices.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `DeviceProfile` dataclass with fields `name: str`, `mem_budget_bytes: int`, `int8_tops: float`, `utilization: float`, `power_w: float`; method `effective_ops_per_s() -> float`. Module constants `ORIN_NANO_8GB` and `ORIN_NX_16GB` (both `DeviceProfile`).

> **Note:** the TOPS/power/mem numbers below are nominal `[EST]` constants — deliberately conservative and tunable. The *math* is tested against synthetic profiles (Task 5); these constants are config, not asserted truth.

- [ ] **Step 1: Write the failing test**

`tests/test_devices.py`:

```python
from edgevla.devices import DeviceProfile, ORIN_NANO_8GB, ORIN_NX_16GB


def test_effective_ops_applies_utilization():
    dev = DeviceProfile(
        name="synthetic",
        mem_budget_bytes=8_000_000_000,
        int8_tops=20.0,
        utilization=0.5,
        power_w=15.0,
    )
    # 20 TOPS * 0.5 = 10e12 ops/s
    assert dev.effective_ops_per_s() == 10e12


def test_estimated_profiles_present_and_ordered():
    assert ORIN_NANO_8GB.mem_budget_bytes < ORIN_NX_16GB.mem_budget_bytes
    assert ORIN_NANO_8GB.name == "orin_nano_8gb"
    assert ORIN_NX_16GB.name == "orin_nx_16gb"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_devices.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'edgevla.devices'`.

- [ ] **Step 3: Write minimal implementation**

`src/edgevla/devices.py`:

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DeviceProfile:
    name: str
    mem_budget_bytes: int
    int8_tops: float
    utilization: float
    power_w: float

    def effective_ops_per_s(self) -> float:
        return self.int8_tops * 1e12 * self.utilization


# Nominal [EST] constants — conservative, tunable. Not measured (no board).
# Usable unified memory is below nameplate (OS + framework reserve).
ORIN_NANO_8GB = DeviceProfile(
    name="orin_nano_8gb",
    mem_budget_bytes=6_500_000_000,   # ~6.5 GB usable of 8 GB
    int8_tops=20.0,                   # conservative dense-INT8 effective
    utilization=0.3,                  # derate for real workloads
    power_w=15.0,
)

ORIN_NX_16GB = DeviceProfile(
    name="orin_nx_16gb",
    mem_budget_bytes=14_000_000_000,  # ~14 GB usable of 16 GB
    int8_tops=70.0,
    utilization=0.3,
    power_w=20.0,
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_devices.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/edgevla/devices.py tests/test_devices.py
git commit -m "feat: DeviceProfile + estimated Orin profiles"
```

---

### Task 5: `three_light_gate` — the 3-light Jetson constraint model

**Files:**
- Create: `src/edgevla/jetson.py`
- Test: `tests/test_jetson.py`

**Interfaces:**
- Consumes: `ModelStats` (Task 3), `DeviceProfile` (Task 4).
- Produces: `GateVerdict` dataclass with fields `fits: bool`, `realtime: bool`, `power_ok: bool`, `peak_mem_gb: float`, `control_rate_hz: float`, `avg_power_w: float`, `energy_j: float`; methods `lights() -> tuple[bool, bool, bool]`, `all_pass() -> bool`. Function `three_light_gate(stats: ModelStats, device: DeviceProfile, *, overhead_bytes: int = 0, rt_floor_hz: float = 10.0, power_budget_w: float = 15.0) -> GateVerdict`.

- [ ] **Step 1: Write the failing test**

`tests/test_jetson.py`:

```python
from edgevla.devices import DeviceProfile
from edgevla.jetson import three_light_gate
from edgevla.modelstats import ModelStats


# Synthetic device: 8 GB budget, 10e12 effective ops/s, 15 W.
DEV = DeviceProfile(
    name="synthetic",
    mem_budget_bytes=8_000_000_000,
    int8_tops=20.0,
    utilization=0.5,
    power_w=15.0,
)


def test_small_fast_model_passes_all_three_lights():
    # peak mem 2 GB; 1e9 flops/step -> 10000 Hz; 15 W <= 15 W
    stats = ModelStats(
        params=10_000_000,
        weight_bytes=1_000_000_000,
        activation_bytes=900_000_000,
        state_bytes=100_000_000,
        flops_per_step=1_000_000_000,
    )
    v = three_light_gate(stats, DEV)
    assert v.lights() == (True, True, True)
    assert v.all_pass() is True
    assert v.peak_mem_gb == 2.0
    assert v.control_rate_hz == 10000.0


def test_fits_but_too_slow_is_the_litevla_edge_failure_mode():
    # peak mem 2 GB (fits); 2e12 flops/step -> 5 Hz (< 10 Hz)
    stats = ModelStats(
        params=7_000_000_000,
        weight_bytes=1_500_000_000,
        activation_bytes=400_000_000,
        state_bytes=100_000_000,
        flops_per_step=2_000_000_000_000,
    )
    v = three_light_gate(stats, DEV)
    assert v.fits is True
    assert v.realtime is False
    assert v.control_rate_hz == 5.0
    assert v.all_pass() is False


def test_over_memory_budget_fails_fits():
    stats = ModelStats(
        params=7_000_000_000,
        weight_bytes=9_000_000_000,
        activation_bytes=0,
        state_bytes=0,
        flops_per_step=1_000_000_000,
    )
    v = three_light_gate(stats, DEV)
    assert v.fits is False


def test_power_budget_can_fail_independently():
    stats = ModelStats(
        params=10_000_000,
        weight_bytes=1_000_000_000,
        activation_bytes=0,
        state_bytes=0,
        flops_per_step=1_000_000_000,
    )
    v = three_light_gate(stats, DEV, power_budget_w=10.0)  # device draws 15 W
    assert v.fits is True
    assert v.realtime is True
    assert v.power_ok is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_jetson.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'edgevla.jetson'`.

- [ ] **Step 3: Write minimal implementation**

`src/edgevla/jetson.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

from edgevla.devices import DeviceProfile
from edgevla.modelstats import ModelStats


@dataclass
class GateVerdict:
    fits: bool
    realtime: bool
    power_ok: bool
    peak_mem_gb: float
    control_rate_hz: float
    avg_power_w: float
    energy_j: float

    def lights(self) -> tuple[bool, bool, bool]:
        return (self.fits, self.realtime, self.power_ok)

    def all_pass(self) -> bool:
        return self.fits and self.realtime and self.power_ok


def three_light_gate(
    stats: ModelStats,
    device: DeviceProfile,
    *,
    overhead_bytes: int = 0,
    rt_floor_hz: float = 10.0,
    power_budget_w: float = 15.0,
) -> GateVerdict:
    peak_mem = stats.peak_mem_bytes(overhead_bytes)
    fits = peak_mem <= device.mem_budget_bytes

    control_rate = device.effective_ops_per_s() / stats.flops_per_step
    realtime = control_rate >= rt_floor_hz

    avg_power = device.power_w
    latency_s = 1.0 / control_rate
    energy = avg_power * latency_s
    power_ok = avg_power <= power_budget_w

    return GateVerdict(
        fits=fits,
        realtime=realtime,
        power_ok=power_ok,
        peak_mem_gb=peak_mem / 1e9,
        control_rate_hz=control_rate,
        avg_power_w=avg_power,
        energy_j=energy,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_jetson.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/edgevla/jetson.py tests/test_jetson.py
git commit -m "feat: 3-light Jetson constraint model"
```

---

### Task 6: `VLAPolicy` seam + `DummyPolicy`

**Files:**
- Create: `src/edgevla/policies/__init__.py`
- Create: `src/edgevla/policies/base.py`
- Create: `src/edgevla/policies/dummy.py`
- Test: `tests/test_policies.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `VLAPolicy` Protocol with attribute `name: str` and methods `reset() -> None`, `act(observation: object) -> numpy.ndarray`. `DummyPolicy(action_dim: int, name: str = "dummy")` implementing it (returns a zero vector of length `action_dim`).

- [ ] **Step 1: Write the failing test**

`tests/test_policies.py`:

```python
import numpy as np

from edgevla.policies.base import VLAPolicy
from edgevla.policies.dummy import DummyPolicy


def test_dummy_policy_is_a_vla_policy():
    p = DummyPolicy(action_dim=7)
    assert isinstance(p, VLAPolicy)  # runtime-checkable Protocol
    assert p.name == "dummy"


def test_dummy_policy_acts_with_correct_shape():
    p = DummyPolicy(action_dim=7)
    p.reset()
    action = p.act(observation={"image": None})
    assert isinstance(action, np.ndarray)
    assert action.shape == (7,)
    assert np.all(action == 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_policies.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'edgevla.policies'`.

- [ ] **Step 3: Write minimal implementation**

`src/edgevla/policies/__init__.py`:

```python
```

`src/edgevla/policies/base.py`:

```python
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class VLAPolicy(Protocol):
    name: str

    def reset(self) -> None: ...

    def act(self, observation: object) -> np.ndarray: ...
```

`src/edgevla/policies/dummy.py`:

```python
from __future__ import annotations

import numpy as np


class DummyPolicy:
    def __init__(self, action_dim: int, name: str = "dummy") -> None:
        self.action_dim = action_dim
        self.name = name

    def reset(self) -> None:
        return None

    def act(self, observation: object) -> np.ndarray:
        return np.zeros(self.action_dim, dtype=np.float32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_policies.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/edgevla/policies tests/test_policies.py
git commit -m "feat: VLAPolicy seam + DummyPolicy"
```

---

### Task 7: `EnvAdapter` seam + `FakeEnv`

**Files:**
- Create: `src/edgevla/envs/__init__.py`
- Create: `src/edgevla/envs/base.py`
- Create: `src/edgevla/envs/fake.py`
- Test: `tests/test_envs_fake.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `EnvAdapter` Protocol with attribute `suites: list[str]` and methods `reset(suite: str | None = None) -> object`, `step(action) -> tuple[object, float, bool, dict]`, `is_success() -> bool`. `FakeEnv(suites: list[str], steps_per_episode: int, success_pattern: list[bool])` implementing it: each `reset()` advances to the next entry of `success_pattern` (cycling); each episode ends after `steps_per_episode` `step()` calls; `is_success()` returns the current episode's scripted outcome.

- [ ] **Step 1: Write the failing test**

`tests/test_envs_fake.py`:

```python
import numpy as np

from edgevla.envs.base import EnvAdapter
from edgevla.envs.fake import FakeEnv


def test_fake_env_is_an_env_adapter():
    env = FakeEnv(suites=["spatial"], steps_per_episode=2, success_pattern=[True])
    assert isinstance(env, EnvAdapter)
    assert env.suites == ["spatial"]


def test_episode_ends_after_steps_and_reports_scripted_success():
    env = FakeEnv(
        suites=["spatial"],
        steps_per_episode=2,
        success_pattern=[True, False],
    )
    # Episode 1 -> success True
    env.reset()
    _, _, done1, _ = env.step(np.zeros(7))
    assert done1 is False
    _, _, done2, _ = env.step(np.zeros(7))
    assert done2 is True
    assert env.is_success() is True
    # Episode 2 -> success False (pattern advances)
    env.reset()
    env.step(np.zeros(7))
    env.step(np.zeros(7))
    assert env.is_success() is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_envs_fake.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'edgevla.envs'`.

- [ ] **Step 3: Write minimal implementation**

`src/edgevla/envs/__init__.py`:

```python
```

`src/edgevla/envs/base.py`:

```python
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EnvAdapter(Protocol):
    suites: list[str]

    def reset(self, suite: str | None = None) -> object: ...

    def step(self, action) -> tuple[object, float, bool, dict]: ...

    def is_success(self) -> bool: ...
```

`src/edgevla/envs/fake.py`:

```python
from __future__ import annotations


class FakeEnv:
    def __init__(
        self,
        suites: list[str],
        steps_per_episode: int,
        success_pattern: list[bool],
    ) -> None:
        self.suites = suites
        self.steps_per_episode = steps_per_episode
        self.success_pattern = success_pattern
        self._episode_idx = -1
        self._step_count = 0
        self._current_success = False

    def reset(self, suite: str | None = None) -> object:
        self._episode_idx += 1
        self._step_count = 0
        self._current_success = self.success_pattern[
            self._episode_idx % len(self.success_pattern)
        ]
        return {"obs": 0.0}

    def step(self, action) -> tuple[object, float, bool, dict]:
        self._step_count += 1
        done = self._step_count >= self.steps_per_episode
        return {"obs": 0.0}, 0.0, done, {}

    def is_success(self) -> bool:
        return self._current_success
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_envs_fake.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/edgevla/envs tests/test_envs_fake.py
git commit -m "feat: EnvAdapter seam + FakeEnv"
```

---

### Task 8: `kpi_assessment` — the KPI-Assessment subroutine

**Files:**
- Create: `src/edgevla/assessment.py`
- Test: `tests/test_assessment.py`

**Interfaces:**
- Consumes: `KPIRow` (Task 2), `ModelStats` (Task 3), `DeviceProfile` (Task 4), `three_light_gate`/`GateVerdict` (Task 5), `VLAPolicy` (Task 6), `EnvAdapter` (Task 7).
- Produces: `kpi_assessment(policy: VLAPolicy, env: EnvAdapter, stats: ModelStats, device: DeviceProfile, *, episodes_per_suite: int, max_steps: int, overhead_bytes: int = 0, rt_floor_hz: float = 10.0, power_budget_w: float = 15.0, peak_mem_gb: float | None = None, disk_size_mb: float | None = None, timer=time.perf_counter) -> tuple[KPIRow, GateVerdict]`. Desktop latency (p50/p95) and `control_rate_hz` come from timing `policy.act`; `success_per_suite`/`success_avg` from rollouts; the `GateVerdict` is the `[EST]` Jetson estimate from `three_light_gate(stats, device)`.

- [ ] **Step 1: Write the failing test**

`tests/test_assessment.py`:

```python
import numpy as np

from edgevla.assessment import kpi_assessment
from edgevla.devices import DeviceProfile
from edgevla.envs.fake import FakeEnv
from edgevla.modelstats import ModelStats
from edgevla.policies.dummy import DummyPolicy


DEV = DeviceProfile(
    name="synthetic",
    mem_budget_bytes=8_000_000_000,
    int8_tops=20.0,
    utilization=0.5,
    power_w=15.0,
)
STATS = ModelStats(
    params=450_000_000,
    weight_bytes=900_000_000,
    activation_bytes=900_000_000,
    state_bytes=200_000_000,
    flops_per_step=1_000_000_000,
)


def make_fake_timer(step_seconds=0.01):
    """Deterministic clock: advances `step_seconds` on every call."""
    t = {"now": 0.0}

    def timer():
        t["now"] += step_seconds
        return t["now"]

    return timer


def test_success_rate_matches_scripted_pattern():
    # 2 episodes/suite, pattern [True, False] -> 50% per suite.
    env = FakeEnv(suites=["spatial"], steps_per_episode=1, success_pattern=[True, False])
    policy = DummyPolicy(action_dim=7)
    row, verdict = kpi_assessment(
        policy, env, STATS, DEV,
        episodes_per_suite=2, max_steps=5,
        peak_mem_gb=2.0, disk_size_mb=900.0,
    )
    assert row.success_per_suite["spatial"] == 0.5
    assert row.success_avg == 0.5


def test_latency_and_control_rate_from_timer():
    env = FakeEnv(suites=["spatial"], steps_per_episode=1, success_pattern=[True])
    policy = DummyPolicy(action_dim=7)
    # Each act() brackets two timer() calls: t0=0.01, t1=0.02 -> 0.01 s = 10 ms.
    row, _ = kpi_assessment(
        policy, env, STATS, DEV,
        episodes_per_suite=1, max_steps=1,
        peak_mem_gb=2.0, disk_size_mb=900.0,
        timer=make_fake_timer(0.01),
    )
    assert round(row.latency_ms_p50, 3) == 10.0
    assert round(row.control_rate_hz, 3) == 100.0  # 1000 / 10 ms


def test_verdict_is_jetson_estimate_and_desktop_row_not_estimated():
    env = FakeEnv(suites=["spatial"], steps_per_episode=1, success_pattern=[True])
    policy = DummyPolicy(action_dim=7)
    row, verdict = kpi_assessment(
        policy, env, STATS, DEV,
        episodes_per_suite=1, max_steps=1,
        peak_mem_gb=2.0, disk_size_mb=900.0,
    )
    assert row.estimated == frozenset()            # desktop numbers measured
    assert verdict.control_rate_hz == 10000.0      # 10e12 ops / 1e9 flops
    assert row.params_m == 450.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_assessment.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'edgevla.assessment'`.

- [ ] **Step 3: Write minimal implementation**

`src/edgevla/assessment.py`:

```python
from __future__ import annotations

import statistics
import time

from edgevla.devices import DeviceProfile
from edgevla.jetson import GateVerdict, three_light_gate
from edgevla.kpi import KPIRow
from edgevla.modelstats import ModelStats


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    frac = k - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def kpi_assessment(
    policy,
    env,
    stats: ModelStats,
    device: DeviceProfile,
    *,
    episodes_per_suite: int,
    max_steps: int,
    overhead_bytes: int = 0,
    rt_floor_hz: float = 10.0,
    power_budget_w: float = 15.0,
    peak_mem_gb: float | None = None,
    disk_size_mb: float | None = None,
    timer=time.perf_counter,
) -> tuple[KPIRow, GateVerdict]:
    success_per_suite: dict[str, float] = {}
    latencies_ms: list[float] = []

    for suite in env.suites:
        successes = 0
        for _ in range(episodes_per_suite):
            env.reset(suite)
            policy.reset()
            for _ in range(max_steps):
                obs = {"suite": suite}
                t0 = timer()
                policy.act(obs)
                t1 = timer()
                latencies_ms.append((t1 - t0) * 1000.0)
                _, _, done, _ = env.step(_zero_like(policy))
                if done:
                    break
            if env.is_success():
                successes += 1
        success_per_suite[suite] = successes / episodes_per_suite

    success_avg = statistics.fmean(success_per_suite.values())
    p50 = _percentile(latencies_ms, 0.50)
    p95 = _percentile(latencies_ms, 0.95)
    control_rate = 1000.0 / p50 if p50 > 0 else 0.0

    if peak_mem_gb is None:
        peak_mem_gb = stats.peak_mem_bytes(overhead_bytes) / 1e9
    if disk_size_mb is None:
        disk_size_mb = stats.weight_bytes / 1e6

    row = KPIRow(
        model_name=getattr(policy, "name", "unknown"),
        device_name=device.name,
        success_avg=success_avg,
        success_per_suite=success_per_suite,
        control_rate_hz=control_rate,
        latency_ms_p50=p50,
        latency_ms_p95=p95,
        peak_mem_gb=peak_mem_gb,
        params_m=stats.params / 1e6,
        disk_size_mb=disk_size_mb,
        estimated=frozenset(),  # desktop-measured row
    )

    verdict = three_light_gate(
        stats,
        device,
        overhead_bytes=overhead_bytes,
        rt_floor_hz=rt_floor_hz,
        power_budget_w=power_budget_w,
    )
    return row, verdict


def _zero_like(policy):
    import numpy as np

    dim = getattr(policy, "action_dim", 1)
    return np.zeros(dim, dtype=np.float32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_assessment.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/edgevla/assessment.py tests/test_assessment.py
git commit -m "feat: KPI-Assessment subroutine"
```

---

### Task 9: Pareto frontier + hypervolume

**Files:**
- Create: `src/edgevla/compression/__init__.py`
- Create: `src/edgevla/compression/pareto.py`
- Test: `tests/test_pareto.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `ParetoPoint` dataclass `(size_mb: float, success: float, label: str)`; `dominates(a: ParetoPoint, b: ParetoPoint) -> bool` (a dominates b iff `a.size_mb <= b.size_mb and a.success >= b.success` and strictly better on at least one); `pareto_frontier(points: list[ParetoPoint]) -> list[ParetoPoint]` (non-dominated, sorted by ascending `size_mb`); `hypervolume(frontier: list[ParetoPoint], ref_size_mb: float, ref_success: float) -> float` (area dominated toward smaller size / higher success, relative to the reference worst-corner).

- [ ] **Step 1: Write the failing test**

`tests/test_pareto.py`:

```python
from edgevla.compression.pareto import (
    ParetoPoint,
    dominates,
    hypervolume,
    pareto_frontier,
)


def test_dominance():
    a = ParetoPoint(size_mb=200, success=0.85, label="a")
    b = ParetoPoint(size_mb=240, success=0.75, label="b")
    assert dominates(a, b) is True
    assert dominates(b, a) is False


def test_frontier_drops_dominated_points():
    pts = [
        ParetoPoint(900, 0.88, "orig"),
        ParetoPoint(450, 0.87, "int8"),
        ParetoPoint(230, 0.82, "int4"),
        ParetoPoint(240, 0.75, "int4+badprune"),  # dominated by int4
    ]
    front = pareto_frontier(pts)
    labels = [p.label for p in front]
    assert "int4+badprune" not in labels
    assert labels == ["int4", "int8", "orig"]  # sorted by size ascending


def test_hypervolume_grows_when_frontier_improves():
    ref_size, ref_success = 1000.0, 0.0
    small = [ParetoPoint(500, 0.80, "a")]
    bigger = [ParetoPoint(500, 0.80, "a"), ParetoPoint(250, 0.82, "b")]
    hv_small = hypervolume(small, ref_size, ref_success)
    hv_big = hypervolume(bigger, ref_size, ref_success)
    assert hv_big > hv_small
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pareto.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'edgevla.compression'`.

- [ ] **Step 3: Write minimal implementation**

`src/edgevla/compression/__init__.py`:

```python
```

`src/edgevla/compression/pareto.py`:

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParetoPoint:
    size_mb: float
    success: float
    label: str


def dominates(a: ParetoPoint, b: ParetoPoint) -> bool:
    no_worse = a.size_mb <= b.size_mb and a.success >= b.success
    strictly_better = a.size_mb < b.size_mb or a.success > b.success
    return no_worse and strictly_better


def pareto_frontier(points: list[ParetoPoint]) -> list[ParetoPoint]:
    front = [
        p for p in points
        if not any(dominates(q, p) for q in points if q is not p)
    ]
    return sorted(front, key=lambda p: p.size_mb)


def hypervolume(
    frontier: list[ParetoPoint],
    ref_size_mb: float,
    ref_success: float,
) -> float:
    """2D hypervolume: minimize size, maximize success.

    Sum of rectangles from each frontier point toward the worst-corner
    reference (ref_size_mb, ref_success). Frontier sorted by ascending size;
    each point covers width (ref_size_mb - size) over the success band above
    the previous (lower-success) point.
    """
    relevant = sorted(
        (p for p in frontier if p.size_mb <= ref_size_mb and p.success >= ref_success),
        key=lambda p: p.success,
    )
    hv = 0.0
    prev_success = ref_success
    for p in relevant:
        width = ref_size_mb - p.size_mb
        height = p.success - prev_success
        hv += width * height
        prev_success = p.success
    return hv
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pareto.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/edgevla/compression/__init__.py src/edgevla/compression/pareto.py tests/test_pareto.py
git commit -m "feat: Pareto frontier + hypervolume"
```

---

### Task 10: Exit criteria (success + diminishing-returns)

**Files:**
- Create: `src/edgevla/compression/exit_criteria.py`
- Test: `tests/test_exit_criteria.py`

**Interfaces:**
- Consumes: `GateVerdict` (Task 5).
- Produces: `success_exit(verdict: GateVerdict, success: float, baseline_success: float, retain: float = 0.90) -> bool` (True iff `verdict.all_pass()` and `success >= retain * baseline_success`); `diminishing_returns_exit(hv_history: list[float], rel_threshold: float = 0.02, window: int = 2) -> bool` (True iff there are more than `window` entries and the relative hypervolume gain over the last `window` rounds is `< rel_threshold`).

- [ ] **Step 1: Write the failing test**

`tests/test_exit_criteria.py`:

```python
from edgevla.compression.exit_criteria import (
    diminishing_returns_exit,
    success_exit,
)
from edgevla.jetson import GateVerdict


def passing_verdict():
    return GateVerdict(
        fits=True, realtime=True, power_ok=True,
        peak_mem_gb=2.0, control_rate_hz=30.0, avg_power_w=10.0, energy_j=0.3,
    )


def failing_verdict():
    return GateVerdict(
        fits=True, realtime=False, power_ok=True,
        peak_mem_gb=2.0, control_rate_hz=6.6, avg_power_w=10.0, energy_j=1.5,
    )


def test_success_exit_needs_all_lights_and_retained_success():
    base = 0.80
    assert success_exit(passing_verdict(), success=0.73, baseline_success=base) is True
    assert success_exit(passing_verdict(), success=0.70, baseline_success=base) is False
    assert success_exit(failing_verdict(), success=0.80, baseline_success=base) is False


def test_diminishing_returns_fires_on_flat_frontier():
    # Big gains early, then plateau within 2%.
    improving = [100.0, 150.0, 220.0]
    assert diminishing_returns_exit(improving) is False
    plateau = [220.0, 222.0, 223.0]  # 223/220 - 1 = 1.36% < 2%
    assert diminishing_returns_exit(plateau) is True


def test_diminishing_returns_needs_enough_history():
    assert diminishing_returns_exit([100.0, 101.0]) is False  # only 2 entries, window=2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_exit_criteria.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'edgevla.compression.exit_criteria'`.

- [ ] **Step 3: Write minimal implementation**

`src/edgevla/compression/exit_criteria.py`:

```python
from __future__ import annotations

from edgevla.jetson import GateVerdict


def success_exit(
    verdict: GateVerdict,
    success: float,
    baseline_success: float,
    retain: float = 0.90,
) -> bool:
    return verdict.all_pass() and success >= retain * baseline_success


def diminishing_returns_exit(
    hv_history: list[float],
    rel_threshold: float = 0.02,
    window: int = 2,
) -> bool:
    if len(hv_history) <= window:
        return False
    past = hv_history[-1 - window]
    current = hv_history[-1]
    if past <= 0:
        return False
    rel_gain = (current - past) / past
    return rel_gain < rel_threshold
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_exit_criteria.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/edgevla/compression/exit_criteria.py tests/test_exit_criteria.py
git commit -m "feat: compression-loop exit criteria"
```

---

### Task 11: `compression_loop` skeleton

**Files:**
- Create: `src/edgevla/compression/loop.py`
- Test: `tests/test_loop.py`

**Interfaces:**
- Consumes: `ParetoPoint`/`pareto_frontier`/`hypervolume` (Task 9), `success_exit`/`diminishing_returns_exit` (Task 10), `GateVerdict` (Task 5).
- Produces: `Variant` dataclass `(point: ParetoPoint, verdict: GateVerdict, success: float)`; `LoopResult` dataclass `(best: ParetoPoint | None, frontier: list[ParetoPoint], hv_history: list[float], rounds: int, exit_reason: str)`; `compression_loop(*, design, sweep, baseline_success: float, ref_size_mb: float, ref_success: float = 0.0, max_rounds: int = 10, retain: float = 0.90, rel_threshold: float = 0.02, window: int = 2) -> LoopResult`. `design(round_idx: int, evidence: dict | None) -> list` returns opaque candidate strategies; `sweep(candidates: list) -> list[Variant]` measures them. The loop implements the §0.5 cycle: design → sweep → update frontier/hypervolume → check exits → pass evidence (last round's variants + frontier) into the next `design`.

- [ ] **Step 1: Write the failing test**

`tests/test_loop.py`:

```python
from edgevla.compression.loop import Variant, compression_loop
from edgevla.compression.pareto import ParetoPoint
from edgevla.jetson import GateVerdict


def passing_verdict():
    return GateVerdict(
        fits=True, realtime=True, power_ok=True,
        peak_mem_gb=2.0, control_rate_hz=30.0, avg_power_w=10.0, energy_j=0.3,
    )


def failing_verdict():
    return GateVerdict(
        fits=True, realtime=False, power_ok=True,
        peak_mem_gb=2.0, control_rate_hz=6.6, avg_power_w=10.0, energy_j=1.5,
    )


def test_loop_exits_on_success_when_a_variant_passes_gates_and_keeps_success():
    def design(round_idx, evidence):
        return ["strategyA"]

    def sweep(candidates):
        # A small, fast model that retains 95% of an 0.80 baseline (0.76 >= 0.72).
        return [Variant(ParetoPoint(250, 0.76, "win"), passing_verdict(), 0.76)]

    result = compression_loop(
        design=design, sweep=sweep,
        baseline_success=0.80, ref_size_mb=1000.0,
    )
    assert result.exit_reason == "success"
    assert result.best.label == "win"
    assert result.rounds == 1


def test_loop_exits_on_diminishing_returns_when_frontier_plateaus():
    # Each round adds a variant that fails real-time (never a success exit),
    # and the Pareto gains shrink below 2% so the loop gives up.
    schedule = [
        ParetoPoint(500, 0.60, "r0"),  # hv0
        ParetoPoint(400, 0.61, "r1"),  # modest gain
        ParetoPoint(399, 0.611, "r2"),  # tiny gain -> plateau
        ParetoPoint(398, 0.612, "r3"),
    ]
    calls = {"i": 0}

    def design(round_idx, evidence):
        return [round_idx]

    def sweep(candidates):
        p = schedule[calls["i"]]
        calls["i"] += 1
        return [Variant(p, failing_verdict(), p.success)]

    result = compression_loop(
        design=design, sweep=sweep,
        baseline_success=0.80, ref_size_mb=1000.0,
        max_rounds=10,
    )
    assert result.exit_reason == "diminishing_returns"
    assert result.best is not None  # best frontier point recorded on give-up


def test_loop_passes_previous_variants_as_evidence():
    seen = []

    def design(round_idx, evidence):
        seen.append(evidence)
        return [round_idx]

    def sweep(candidates):
        return [Variant(ParetoPoint(500, 0.50, "x"), failing_verdict(), 0.50)]

    compression_loop(
        design=design, sweep=sweep,
        baseline_success=0.80, ref_size_mb=1000.0,
        max_rounds=2,
    )
    assert seen[0] is None                       # round 0: no evidence yet
    assert "variants" in seen[1]                 # round 1: prior variants fed back
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_loop.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'edgevla.compression.loop'`.

- [ ] **Step 3: Write minimal implementation**

`src/edgevla/compression/loop.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

from edgevla.compression.exit_criteria import (
    diminishing_returns_exit,
    success_exit,
)
from edgevla.compression.pareto import ParetoPoint, hypervolume, pareto_frontier
from edgevla.jetson import GateVerdict


@dataclass
class Variant:
    point: ParetoPoint
    verdict: GateVerdict
    success: float


@dataclass
class LoopResult:
    best: ParetoPoint | None
    frontier: list[ParetoPoint]
    hv_history: list[float]
    rounds: int
    exit_reason: str


def _best_on_frontier(frontier: list[ParetoPoint]) -> ParetoPoint | None:
    if not frontier:
        return None
    return max(frontier, key=lambda p: p.success)


def compression_loop(
    *,
    design,
    sweep,
    baseline_success: float,
    ref_size_mb: float,
    ref_success: float = 0.0,
    max_rounds: int = 10,
    retain: float = 0.90,
    rel_threshold: float = 0.02,
    window: int = 2,
) -> LoopResult:
    frontier: list[ParetoPoint] = []
    hv_history: list[float] = []
    evidence: dict | None = None

    for round_idx in range(max_rounds):
        candidates = design(round_idx, evidence)
        variants = sweep(candidates)

        frontier = pareto_frontier(frontier + [v.point for v in variants])
        hv_history.append(hypervolume(frontier, ref_size_mb, ref_success))

        for v in variants:
            if success_exit(v.verdict, v.success, baseline_success, retain):
                return LoopResult(
                    best=v.point,
                    frontier=frontier,
                    hv_history=hv_history,
                    rounds=round_idx + 1,
                    exit_reason="success",
                )

        if diminishing_returns_exit(hv_history, rel_threshold, window):
            return LoopResult(
                best=_best_on_frontier(frontier),
                frontier=frontier,
                hv_history=hv_history,
                rounds=round_idx + 1,
                exit_reason="diminishing_returns",
            )

        evidence = {"variants": variants, "frontier": frontier}

    return LoopResult(
        best=_best_on_frontier(frontier),
        frontier=frontier,
        hv_history=hv_history,
        rounds=max_rounds,
        exit_reason="max_rounds",
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_loop.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest -v`
Expected: PASS (all tests from Tasks 1–11 green).

- [ ] **Step 6: Commit**

```bash
git add src/edgevla/compression/loop.py tests/test_loop.py
git commit -m "feat: evidence-driven compression-loop skeleton"
```

---

### Task 12: LIBERO + `vla-evaluation-harness` integration smoke

> This task is **verified by running**, not by unit tests — it wires the real sim behind the `EnvAdapter`/`VLAPolicy` seams. The harness/LIBERO APIs vary by version: **read the installed package's actual entry points before writing `libero.py`; do not assume function signatures.** If the `vla-evaluation-harness` LIBERO bridge is not usable as-installed, the documented fallback is LIBERO's own evaluation scripts wrapped behind the same `EnvAdapter` interface.

**Files:**
- Create: `src/edgevla/envs/libero.py`
- Create: `scripts/smoke_libero.py`
- Modify: `pyproject.toml` (add an optional `sim` dependency group)

**Interfaces:**
- Consumes: `EnvAdapter` (Task 7), `DummyPolicy` (Task 6), `kpi_assessment` (Task 8), `ModelStats` (Task 3), `ORIN_NANO_8GB` (Task 4).
- Produces: `LiberoEnv(suite_names: list[str])` implementing `EnvAdapter` over LIBERO via the installed runtime; `scripts/smoke_libero.py` that runs `DummyPolicy` through `kpi_assessment` on one real LIBERO suite and prints a `KPIRow` + 3-light verdict.

- [ ] **Step 1: Add the sim dependency group**

Edit `pyproject.toml`, add under `[project.optional-dependencies]`:

```toml
sim = ["LIBERO", "vla-evaluation-harness"]
```

- [ ] **Step 2: Install LIBERO + the harness and confirm import**

Run:
```bash
python -m pip install -e ".[sim]"
python -c "import libero; print('libero ok')"
```
Expected: prints `libero ok`. If install fails on Windows, install under WSL2/Linux on the 4090 host (per README platform caveat) and continue there.

- [ ] **Step 3: Inspect the real API before coding the adapter**

Run:
```bash
python -c "import libero.libero.benchmark as b; print([m for m in dir(b) if not m.startswith('_')])"
```
Record the actual benchmark/suite accessors and the env factory. Use the names you observe in Step 4 — do not invent them.

- [ ] **Step 4: Implement `LiberoEnv` against the observed API**

`src/edgevla/envs/libero.py` (adapt the calls in `reset`/`step`/`is_success` to the names recorded in Step 3; the structure below is fixed, the wrapped calls are filled from the real API):

```python
from __future__ import annotations


class LiberoEnv:
    """EnvAdapter over LIBERO. Wraps the installed runtime's env object.

    The three private hooks (_make_env, _read_done, _read_success) are the only
    version-specific parts; fill them from the API inspected in Step 3.
    """

    def __init__(self, suite_names: list[str]) -> None:
        self.suites = suite_names
        self._env = None
        self._last_info: dict = {}
        self._success = False

    def _make_env(self, suite: str):
        # Fill from Step 3's observed factory, e.g. benchmark + env builder.
        raise NotImplementedError("wire to installed LIBERO env factory")

    def reset(self, suite: str | None = None) -> object:
        suite = suite or self.suites[0]
        self._env = self._make_env(suite)
        self._success = False
        return self._env.reset()

    def step(self, action) -> tuple[object, float, bool, dict]:
        obs, reward, done, info = self._env.step(action)
        self._last_info = info
        if done:
            self._success = bool(info.get("success", reward > 0))
        return obs, reward, done, info

    def is_success(self) -> bool:
        return self._success
```

- [ ] **Step 5: Write the smoke script**

`scripts/smoke_libero.py`:

```python
"""Smoke test: DummyPolicy through kpi_assessment on one real LIBERO suite."""
from __future__ import annotations

from edgevla.assessment import kpi_assessment
from edgevla.devices import ORIN_NANO_8GB
from edgevla.envs.libero import LiberoEnv
from edgevla.modelstats import ModelStats
from edgevla.policies.dummy import DummyPolicy


def main() -> None:
    env = LiberoEnv(suite_names=["libero_spatial"])
    policy = DummyPolicy(action_dim=7)
    stats = ModelStats(
        params=450_000_000,
        weight_bytes=900_000_000,
        activation_bytes=900_000_000,
        state_bytes=200_000_000,
        flops_per_step=1_000_000_000,
    )
    row, verdict = kpi_assessment(
        policy, env, stats, ORIN_NANO_8GB,
        episodes_per_suite=2, max_steps=200,
        peak_mem_gb=2.0, disk_size_mb=900.0,
    )
    print("KPI:", row.to_dict())
    print("Jetson 3-light [EST]:", verdict.lights())


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the smoke test**

Run:
```bash
python scripts/smoke_libero.py
```
Expected: the LIBERO loop runs to completion and prints a `KPI: {...}` dict (DummyPolicy success near 0.0 is fine — the point is the loop closes) and `Jetson 3-light [EST]: (True, True, True)`. If it raises from the `_make_env`/`step` wiring, fix `libero.py` against the real API (Step 3) until the loop closes.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/edgevla/envs/libero.py scripts/smoke_libero.py
git commit -m "feat: LIBERO EnvAdapter + closed-loop smoke run"
```

---

## Self-Review

**1. Spec coverage (masterplan §0.1–§0.5):**
- §0.1 environment & sim setup → Task 1 (package/pytest), Task 12 (LIBERO + harness install, smoke loop). ✅
- §0.2 KPI glossary → Task 2 (`KPIRow` carries every glossary field; `estimated` set marks `[EST]`). ✅
- §0.3 3-light Jetson model → Task 4 (profiles) + Task 5 (`three_light_gate`, incl. the "fits but too slow" test). ✅
- §0.4 reusable subroutines → Task 8 (`kpi_assessment`), Tasks 9–11 (`compression_loop` + machinery). ✅
- §0.5 evidence-driven loop (design→sweep→analyze→synthesize→exit) → Task 11 loop with `design`/`sweep` callbacks and evidence feedback; Tasks 9–10 supply Pareto + dual exit. ✅
- Decoupled `VLAPolicy`/`EnvAdapter` seams with Dummy/Fake backends → Tasks 6, 7. ✅

**2. Placeholder scan:** No "TBD/handle edge cases/similar to Task N." Task 12's `_make_env` is an explicit `NotImplementedError` the engineer fills from the *observed* real API — that is the intended seam for version-specific code, not a placeholder for logic this plan owns. ✅

**3. Type consistency:** `ModelStats(params, weight_bytes, activation_bytes, state_bytes, flops_per_step)` and `peak_mem_bytes()` are used identically in Tasks 3/5/8/12. `GateVerdict.all_pass()`/`.lights()` used consistently in Tasks 5/10/11. `ParetoPoint(size_mb, success, label)` and `Variant(point, verdict, success)` consistent across Tasks 9/11. `kpi_assessment(...)` signature matches its call in Task 12. ✅

---

## Notes for the implementer
- Run `python -m pytest -v` after each task; the suite must stay green.
- Tasks 1–11 need only `numpy` + `pytest` (no GPU, no sim) — they are the verified core.
- Task 12 is the only step that needs the 4090 + LIBERO; if Windows install is painful, do it under WSL2/Linux per the README platform caveat.
- The device-profile constants in Task 4 are `[EST]` and meant to be tuned as better Jetson data appears — only the *math* (Task 5) is asserted.
