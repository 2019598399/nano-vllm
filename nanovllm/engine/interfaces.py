from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto

from nanovllm.engine.sequence import Sequence


class ExecutionPhase(Enum):
    PREFILL = auto()
    DECODE = auto()


@dataclass
class SchedulerOutput:
    sequences: list[Sequence]
    phase: ExecutionPhase
    num_tokens: int
    preempted: list[Sequence] = field(default_factory=list)


@dataclass
class EngineStepOutput:
    finished: list[tuple[int, list[int]]]
    num_tokens: int
    token_events: list["TokenEvent"] = field(default_factory=list)


@dataclass(frozen=True)
class TokenEvent:
    seq_id: int
    token_id: int
    timestamp: float
    phase: ExecutionPhase


class EngineProxy(ABC):
    @abstractmethod
    def add(self, seq: Sequence):
        raise NotImplementedError

    @abstractmethod
    def step(self) -> EngineStepOutput:
        raise NotImplementedError

    @abstractmethod
    def is_finished(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError


class BaseScheduler(ABC):
    @abstractmethod
    def add(self, seq: Sequence):
        raise NotImplementedError

    @abstractmethod
    def schedule(self) -> SchedulerOutput:
        raise NotImplementedError

    @abstractmethod
    def is_finished(self) -> bool:
        raise NotImplementedError


class BaseModelRunner(ABC):
    @abstractmethod
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        raise NotImplementedError
