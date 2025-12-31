import os
from dataclasses import dataclass

from libero.libero import benchmark, get_libero_path

from src.opentau.utils.monkey_patch import torch_load_patch
from src.opentau.configs.train import TrainPipelineConfig

LIBERO_BENCHMARK_DICT = benchmark.get_benchmark_dict()


@dataclass
class LiberoEnvConfig:
    suite: str  # Task suite to run. Must be 'spatial', 'object', 'goal', or '100'.
    id: int  # index of the task in the suite to run.
    max_steps: int = 1000  # maximum number of steps to run for each task.
    chunk_usage: int | None = (
        None  # number of actions to perform in each chunk before getting a new observation.
    )
    n_simulations: int = 100  # number of simulations to run for each task.
    video_dir: str = None  # directory to save videos of the task execution.

    def __post_init__(self):
        torch_load_patch()
        suite = f"libero_{self.suite}".lower()
        if suite not in LIBERO_BENCHMARK_DICT:
            raise ValueError(
                f"Invalid suites: '{self.suite}'. "
                f"Available suites are: {[k.replace('libero_', '') for k in LIBERO_BENCHMARK_DICT]}"
            )
        suite = LIBERO_BENCHMARK_DICT[suite]()
        try:
            task = suite.get_task(self.id)
        except IndexError as e:
            raise ValueError(
                f"Invalid task id: {self.id} for suite: {self.suite}. "
                f"Available ids must be from 0 to {len(suite.tasks) - 1}."
            ) from e

        self.bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        self.init_states = suite.get_task_init_states(self.id)
        self.task = task


@dataclass
class TrainConfigWithLiberoEval(TrainPipelineConfig):
    libero: LiberoEnvConfig = None

    def __post_init__(self):
        super().__post_init__()
        if self.libero is None:
            raise ValueError("Libero config must be provided.")
        if self.libero.chunk_usage is None:
            self.libero.chunk_usage = self.action_chunk
        assert 1 <= self.libero.chunk_usage <= self.action_chunk, (
            f"Chunk usage must be between 1 and {self.action_chunk=}, got {self.libero.chunk_usage=}."
        )
