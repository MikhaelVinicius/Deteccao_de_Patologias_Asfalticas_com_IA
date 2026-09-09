from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from roadedge.common import (
    resolve_path,
    sha256_file,
    write_json,
)


def metric_value(
    metrics: object,
    attribute: str,
) -> float:
    value = metrics

    for part in attribute.split("."):
        value = getattr(
            value,
            part,
        )

    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Avalia uma unica vez no teste congelado."
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
        "--device",
        default="cpu",
    )

    parser.add_argument(
        "--split",
        choices=[
            "val",
            "test",
        ],
        default="test",
    )

    parser.add_argument(
        "--output",
        default="artifacts/evaluation",
    )

    args = parser.parse_args()

    model_path = (
        resolve_path(args.model)
        if Path(args.model).exists()
        else Path(args.model)
    )

    model = YOLO(
        str(model_path)
    )

    result = model.val(
        data=str(
            resolve_path(
                args.data
            )
        ),
        split=args.split,
        imgsz=args.imgsz,
        device=args.device,
        conf=0.001,
        iou=0.70,
        max_det=300,
        plots=True,
        save_json=True,
        verbose=True,
    )

    speed = {
        key: float(value)
        for key, value
        in result.speed.items()
    }

    payload = {
        "model": str(
            model_path
        ),
        "model_sha256": (
            sha256_file(model_path)
            if model_path.exists()
            else "official_checkpoint"
        ),
        "split": args.split,
        "image_size": args.imgsz,
        "precision": metric_value(
            result,
            "box.mp",
        ),
        "recall": metric_value(
            result,
            "box.mr",
        ),
        "map50": metric_value(
            result,
            "box.map50",
        ),
        "map50_95": metric_value(
            result,
            "box.map",
        ),
        "map50_95_per_class": np.asarray(
            result.box.maps,
            dtype=float,
        ).tolist(),
        "speed_ms": speed,
        "note": (
            "AP usa conf=0.001; o limiar de producao "
            "deve ser escolhido somente na validacao."
        ),
    }

    output = (
        resolve_path(args.output)
        / f"{model_path.stem}_{args.split}.json"
    )

    write_json(
        output,
        payload,
    )

    print(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
