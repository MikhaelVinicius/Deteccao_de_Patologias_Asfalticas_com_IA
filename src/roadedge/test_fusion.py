from roadedge.fusion import (
    FusionEngine,
    ImpactEvent,
    VisualEvent,
)


def visual(**updates):
    values = dict(
        event_id="v1",
        timestamp_s=100.0,
        latitude=-8.8900,
        longitude=-36.4900,
        class_name="panela",
        confidence=0.80,
        in_driving_roi=True,
    )

    values.update(
        updates
    )

    return VisualEvent(
        **values
    )


def test_confirms_temporally_and_spatially_compatible_impact():
    impact = ImpactEvent(
        "i1",
        101.2,
        -8.8901,
        -36.4901,
        0.90,
    )

    result = (
        FusionEngine().fuse(
            visual(),
            [impact],
        )
    )

    assert (
        result.status
        == "confirmed"
    )

    assert (
        result.impact_id
        == "i1"
    )


def test_keeps_visual_evidence_when_vehicle_does_not_hit_damage():
    result = (
        FusionEngine().fuse(
            visual(),
            [],
        )
    )

    assert (
        result.status
        == "visual_only"
    )


def test_rejects_detection_outside_driving_corridor():
    result = (
        FusionEngine().fuse(
            visual(
                in_driving_roi=False
            ),
            [],
        )
    )

    assert (
        result.status
        == "rejected_visual"
    )
