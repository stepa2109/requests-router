"""Сервис маршрутизации: модель + правило уверенности + срочность."""
from dataclasses import dataclass
from pathlib import Path

from sklearn.pipeline import Pipeline

from router import model
from router.config import (CONFIDENCE_THRESHOLD, MARKER_SECOND_OPTION_THRESHOLD, MIN_TEXT_LEN, MODEL_PATH,
                           SECOND_OPTION_THRESHOLD)
from router.multi_topic import has_extra_question
from router.preprocessing import normalize
from router.urgency import detect_urgency


@dataclass
class RoutingResult:
    department: str
    confidence: float
    urgency: str
    auto_routed: bool
    top3: list[tuple[str, float]]
    multi_topic: bool = False


class RequestRouter:
    def __init__(self, pipeline: Pipeline, threshold: float = CONFIDENCE_THRESHOLD,
                 second_threshold: float = SECOND_OPTION_THRESHOLD,
                 marker_second_threshold: float = MARKER_SECOND_OPTION_THRESHOLD):
        self.pipeline = pipeline
        self.threshold = threshold
        self.second_threshold = second_threshold
        self.marker_second_threshold = marker_second_threshold

    @classmethod
    def from_disk(cls, path: Path = MODEL_PATH) -> "RequestRouter":
        return cls(model.load(path))

    def route(self, text: str) -> RoutingResult:
        if len(normalize(text)) < MIN_TEXT_LEN:
            raise ValueError("Текст обращения слишком короткий")
        probabilities = self.pipeline.predict_proba([text])[0]
        ranked = sorted(zip(self.pipeline.classes_, probabilities), key=lambda p: p[1], reverse=True)
        department, confidence = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        second_threshold = min(self.second_threshold,
                               self.marker_second_threshold if has_extra_question(text) else 1.0)
        multi_topic = second >= second_threshold
        return RoutingResult(
            department=str(department),
            confidence=float(confidence),
            urgency=detect_urgency(text),
            auto_routed=bool(confidence >= self.threshold and not multi_topic),
            top3=[(str(d), float(p)) for d, p in ranked[:3]],
            multi_topic=bool(multi_topic),
        )
