from __future__ import annotations

from dataclasses import (
    asdict,
    dataclass,
)
from math import (
    asin,
    cos,
    radians,
    sin,
    sqrt,
)


@dataclass(frozen=True)
class VisualEvent:
    event_id: str
    timestamp_s: float
    latitude: float
    longitude: float
    class_name: str
    confidence: float
    in_driving_roi: bool


@dataclass(frozen=True)
class ImpactEvent:
    event_id: str
    timestamp_s: float
    latitude: float
    longitude: float
    probability: float


@dataclass(frozen=True)
class FusedEvent:
    visual_id: str
    impact_id: str | None
    status: str
    score: float
    time_delta_s: float | None
    distance_m: float | None

    def to_dict(
        self,
    ) -> dict:
        return asdict(
            self
        )


def haversine_m(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    earth_radius_m = (
        6_371_000.0
    )

    dlat = radians(
        lat2 - lat1
    )

    dlon = radians(
        lon2 - lon1
    )

    a = (
        sin(dlat / 2) ** 2
        + cos(radians(lat1))
        * cos(radians(lat2))
        * sin(dlon / 2) ** 2
    )

    return (
        2
        * earth_radius_m
        * asin(
            sqrt(a)
        )
    )


class FusionEngine:
    def __init__(
        self,
        min_delay_s: float = 0.10,
        max_delay_s: float = 3.00,
        max_distance_m: float = 30.0,
        min_visual_confidence: float = 0.25,
        min_impact_probability: float = 0.60,
    ) -> None:
        self.min_delay_s = (
            min_delay_s
        )

        self.max_delay_s = (
            max_delay_s
        )

        self.max_distance_m = (
            max_distance_m
        )

        self.min_visual_confidence = (
            min_visual_confidence
        )

        self.min_impact_probability = (
            min_impact_probability
        )

    def fuse(
        self,
        visual: VisualEvent,
        impacts: list[ImpactEvent],
    ) -> FusedEvent:
        if (
            visual.confidence
            < self.min_visual_confidence
            or not visual.in_driving_roi
        ):
            return FusedEvent(
                visual.event_id,
                None,
                "rejected_visual",
                visual.confidence,
                None,
                None,
            )

        candidates: list[
            tuple[
                float,
                ImpactEvent,
                float,
                float,
            ]
        ] = []

        for impact in impacts:
            delta = (
                impact.timestamp_s
                - visual.timestamp_s
            )

            if not (
                self.min_delay_s
                <= delta
                <= self.max_delay_s
            ):
                continue

            distance = haversine_m(
                visual.latitude,
                visual.longitude,
                impact.latitude,
                impact.longitude,
            )

            if (
                distance
                > self.max_distance_m
                or impact.probability
                < self.min_impact_probability
            ):
                continue

            time_score = (
                1.0
                - (
                    delta
                    - self.min_delay_s
                )
                / (
                    self.max_delay_s
                    - self.min_delay_s
                )
            )

            distance_score = (
                1.0
                - distance
                / self.max_distance_m
            )

            score = (
                0.50
                * visual.confidence
                + 0.30
                * impact.probability
                + 0.10
                * time_score
                + 0.10
                * distance_score
            )

            candidates.append(
                (
                    score,
                    impact,
                    delta,
                    distance,
                )
            )

        if not candidates:
            return FusedEvent(
                visual.event_id,
                None,
                "visual_only",
                visual.confidence,
                None,
                None,
            )

        (
            score,
            impact,
            delta,
            distance,
        ) = max(
            candidates,
            key=lambda item: item[0],
        )

        return FusedEvent(
            visual.event_id,
            impact.event_id,
            "confirmed",
            score,
            delta,
            distance,
        )
