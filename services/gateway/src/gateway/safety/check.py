from typing import Literal

from gateway.backends.base import Message
from gateway.config import Settings
from gateway.safety.detectors import Detector, HeuristicDetector, TfidfDetector, paragraph_score

SafetyMode = Literal["off", "flag", "block"]


class InjectionCheck:
    """Scores the user messages of a request for prompt injection (D-017).

    Only `user` messages are scanned. Contract for callers: untrusted text (issue bodies,
    retrieved documents) goes in user messages; system messages are the caller's own
    instructions and would otherwise be flagged constantly ("ignore...", "you are...").
    """

    def __init__(self, detector: Detector, threshold: float, mode: SafetyMode) -> None:
        self.detector = detector
        self.threshold = threshold
        self.mode = mode

    def score(self, messages: tuple[Message, ...]) -> float:
        return max(
            (paragraph_score(self.detector, m.content) for m in messages if m.role == "user"),
            default=0.0,
        )


def build_injection_check(settings: Settings) -> InjectionCheck | None:
    if settings.safety_mode == "off":
        return None
    if settings.safety_detector == "tfidf":
        detector: Detector = TfidfDetector.load(settings.safety_model_path)
    else:
        detector = HeuristicDetector()
    return InjectionCheck(detector, settings.safety_threshold, settings.safety_mode)
