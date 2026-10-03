import numpy as np
from sklearn.ensemble import RandomForestClassifier

from roadedge.forest_json import (
    export_forest,
    load_payload,
    predict_proba_json,
)


def test_exported_forest_preserves_class_prediction(
    tmp_path,
):
    x = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.2],
            [1.0, 1.0],
            [0.9, 0.8],
        ]
    )

    y = np.asarray(
        [
            0,
            0,
            1,
            1,
        ]
    )

    model = RandomForestClassifier(
        n_estimators=7,
        max_depth=3,
        random_state=42,
    ).fit(
        x,
        y,
    )

    target = (
        tmp_path
        / "forest.json"
    )

    export_forest(
        model,
        [
            "a",
            "b",
        ],
        target,
    )

    payload = load_payload(
        target
    )

    for row in x:
        expected = model.predict(
            row.reshape(
                1,
                -1,
            )
        )[0]

        actual = payload[
            "classes"
        ][
            predict_proba_json(
                payload,
                {
                    "a": row[0],
                    "b": row[1],
                },
            ).argmax()
        ]

        assert (
            actual
            == expected
        )
