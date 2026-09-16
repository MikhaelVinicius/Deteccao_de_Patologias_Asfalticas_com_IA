from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier


def export_forest(
    model: RandomForestClassifier,
    feature_names: list[str],
    path: Path,
) -> None:
    trees: list[
        dict[str, Any]
    ] = []

    for estimator in model.estimators_:
        tree = estimator.tree_

        trees.append(
            {
                "feature": (
                    tree.feature.tolist()
                ),
                "threshold": (
                    tree.threshold.tolist()
                ),
                "left": (
                    tree.children_left.tolist()
                ),
                "right": (
                    tree.children_right.tolist()
                ),
                "value": (
                    tree.value[
                        :, 0, :
                    ].tolist()
                ),
            }
        )

    payload = {
        "format_version": 1,
        "classes": (
            model.classes_.tolist()
        ),
        "features": feature_names,
        "trees": trees,
    }

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )


def predict_proba_json(
    payload: dict[str, Any],
    row: dict[str, float],
) -> np.ndarray:
    vector = np.asarray(
        [
            row[name]
            for name
            in payload["features"]
        ],
        dtype=float,
    )

    probabilities = np.zeros(
        len(
            payload["classes"]
        ),
        dtype=float,
    )

    for tree in payload["trees"]:
        node = 0

        while (
            tree["left"][node]
            != tree["right"][node]
        ):
            feature = (
                tree["feature"][node]
            )

            node = (
                tree["left"][node]
                if (
                    vector[feature]
                    <= tree["threshold"][node]
                )
                else tree["right"][node]
            )

        counts = np.asarray(
            tree["value"][node],
            dtype=float,
        )

        probabilities += (
            counts
            / max(
                counts.sum(),
                1.0,
            )
        )

    return (
        probabilities
        / len(
            payload["trees"]
        )
    )


def load_payload(
    path: Path,
) -> dict[str, Any]:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )
