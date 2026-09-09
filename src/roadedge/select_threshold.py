from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from ultralytics import YOLO

from roadedge.common import resolve_path, write_json


def xywh_to_xyxy(box: list[float]) -> np.ndarray:
    x, y, width, height = box

    return np.asarray(
        [
            x - width / 2,
            y - height / 2,
            x + width / 2,
            y + height / 2,
        ]
    )


def iou(
    one: np.ndarray,
    two: np.ndarray,
) -> float:
    x1, y1 = np.maximum(
        one[:2],
        two[:2],
    )

    x2, y2 = np.minimum(
        one[2:],
        two[2:],
    )

    intersection = (
        max(0.0, x2 - x1)
        * max(0.0, y2 - y1)
    )

    area_one = (
        max(0.0, one[2] - one[0])
        * max(0.0, one[3] - one[1])
    )

    area_two = (
        max(0.0, two[2] - two[0])
        * max(0.0, two[3] - two[1])
    )

    return intersection / max(
        area_one + area_two - intersection,
        1e-12,
    )


def ground_truth(
    image_path: Path,
) -> list[tuple[int, np.ndarray]]:
    parts = list(
        image_path.parts
    )

    if "images" not in parts:
        raise ValueError(
            f"O caminho deve conter uma pasta images: "
            f"{image_path}"
        )

    parts[
        parts.index("images")
    ] = "labels"

    label_path = Path(
        *parts
    ).with_suffix(".txt")

    boxes: list[
        tuple[int, np.ndarray]
    ] = []

    if label_path.exists():
        for line in (
            label_path
            .read_text()
            .splitlines()
        ):
            values = line.split()

            if len(values) == 5:
                boxes.append(
                    (
                        int(values[0]),
                        xywh_to_xyxy(
                            list(
                                map(
                                    float,
                                    values[1:],
                                )
                            )
                        ),
                    )
                )

    return boxes


def match_counts(
    truth: list[
        tuple[int, np.ndarray]
    ],
    predictions: list[
        tuple[
            int,
            float,
            np.ndarray,
        ]
    ],
    confidence: float,
    iou_threshold: float,
) -> tuple[int, int, int]:
    selected = sorted(
        (
            item
            for item in predictions
            if item[1] >= confidence
        ),
        key=lambda x: -x[1],
    )

    matched: set[int] = set()

    true_positives = 0

    for class_id, _, box in selected:
        candidates = [
            (
                index,
                iou(
                    box,
                    target_box,
                ),
            )
            for index, (
                target_class,
                target_box,
            ) in enumerate(truth)
            if (
                target_class == class_id
                and index not in matched
            )
        ]

        if candidates:
            best_index, best_iou = max(
                candidates,
                key=lambda pair: pair[1],
            )

            if best_iou >= iou_threshold:
                matched.add(
                    best_index
                )

                true_positives += 1

    false_positives = (
        len(selected)
        - true_positives
    )

    false_negatives = (
        len(truth)
        - true_positives
    )

    return (
        true_positives,
        false_positives,
        false_negatives,
    )


def prf(
    tp: int,
    fp: int,
    fn: int,
) -> dict[str, float]:
    precision = (
        tp
        / max(tp + fp, 1)
    )

    recall = (
        tp
        / max(tp + fn, 1)
    )

    f1 = (
        2
        * precision
        * recall
        / max(
            precision + recall,
            1e-12,
        )
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def split_paths(
    data_yaml: Path,
    split: str,
) -> list[Path]:
    config = yaml.safe_load(
        data_yaml.read_text(
            encoding="utf-8"
        )
    )

    root = (
        data_yaml.parent
        / config.get(
            "path",
            ".",
        )
    )

    list_path = (
        root
        / config[split]
    ).resolve()

    return [
        Path(line)
        for line
        in list_path
        .read_text()
        .splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Escolhe conf na validacao, "
            "nunca no teste."
        )
    )

    parser.add_argument(
        "--model",
        required=True,
    )

    parser.add_argument(
        "--data",
        default="configs/dataset.yaml",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--output",
        default="artifacts/thresholds",
    )

    args = parser.parse_args()

    data_yaml = resolve_path(
        args.data
    )

    images = split_paths(
        data_yaml,
        "val",
    )

    model_path = resolve_path(
        args.model
    )

    model = YOLO(
        str(model_path)
    )

    cached: list[
        tuple[
            list[
                tuple[
                    int,
                    np.ndarray,
                ]
            ],
            list[
                tuple[
                    int,
                    float,
                    np.ndarray,
                ]
            ],
        ]
    ] = []

    for image_path, result in zip(
        images,
        model.predict(
            [
                str(path)
                for path in images
            ],
            conf=0.001,
            imgsz=args.imgsz,
            stream=True,
            verbose=False,
        ),
        strict=True,
    ):
        predictions = [
            (
                int(
                    box.cls.item()
                ),
                float(
                    box.conf.item()
                ),
                box.xyxyn
                .cpu()
                .numpy()[0],
            )
            for box in result.boxes
        ]

        cached.append(
            (
                ground_truth(
                    image_path
                ),
                predictions,
            )
        )

    table: list[
        dict[str, float]
    ] = []

    for threshold in np.linspace(
        0.05,
        0.80,
        76,
    ):
        totals = np.zeros(
            3,
            dtype=int,
        )

        for truth, predictions in cached:
            totals += np.asarray(
                match_counts(
                    truth,
                    predictions,
                    float(
                        threshold
                    ),
                    args.iou,
                )
            )

        table.append(
            {
                "threshold": float(
                    threshold
                ),
                **prf(
                    *map(
                        int,
                        totals,
                    )
                ),
            }
        )

    best = max(
        table,
        key=lambda row: (
            row["f1"],
            row["recall"],
        ),
    )

    payload = {
        "model": str(
            model_path
        ),
        "selected_on": "validation",
        "matching_iou": args.iou,
        "best": best,
        "sweep": table,
        "instruction": (
            "Congele este limiar antes "
            "de executar o teste final."
        ),
    }

    output = (
        resolve_path(
            args.output
        )
        / (
            f"{model_path.stem}"
            "_threshold.json"
        )
    )

    write_json(
        output,
        payload,
    )

    print(
        json.dumps(
            best,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
