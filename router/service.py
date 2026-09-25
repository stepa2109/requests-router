"""Сервис маршрутизации: модель + правило уверенности + срочность."""
from dataclasses import dataclass
from pathlib import Path

from sklearn.pipeline import Pipeline

from router import model
from router.config import CONFIDENCE_THRESHOLD, MIN_TEXT_LEN, MODEL_PATH
from router.preprocessing import normalize
from router.urgency import detect_urgency


@dataclass
class RoutingResult:
    department: str
    confidence: float
    urgency: str
    auto_routed: bool
    top3: list[tuple[str, float]]


class RequestRouter:
    def __init__(self, pipeline: Pipeline, threshold: float = CONFIDENCE_THRESHOLD):
        self.pipeline = pipeline
        self.threshold = threshold

    @classmethod
    def from_disk(cls, path: Path = MODEL_PATH) -> "RequestRouter":
        return cls(model.load(path))

    def route(self, text: str) -> RoutingResult:
        if len(normalize(text)) < MIN_TEXT_LEN:
            raise ValueError("Текст обращения слишком короткий")
        probabilities = self.pipeline.predict_proba([text])[0]
        ranked = sorted(zip(self.pipeline.classes_, probabilities), key=lambda p: p[1], reverse=True)
        department, confidence = ranked[0]
        return RoutingResult(
            department=str(department),
            confidence=float(confidence),
            urgency=detect_urgency(text),
            auto_routed=bool(confidence >= self.threshold),
            top3=[(str(d), float(p)) for d, p in ranked[:3]],
        )
