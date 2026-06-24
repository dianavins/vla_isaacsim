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
