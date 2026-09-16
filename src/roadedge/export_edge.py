from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from ultralytics import YOLO

from roadedge.common import (
    resolve_path,
    sha256_file,
    write_json,
)


def copy_export(
    exported: str | Path,
    target: Path,
) -> Path:
    source = Path(exported)

    if source.is_dir():
        candidates = sorted(
            source.rglob("*.tflite")
        )

        if not candidates:
            raise FileNotFoundError(
                f"Nenhum .tflite encontrado em {source}"
            )

        source = candidates[0]

    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        source,
        target,
    )

    return target


def export_litert(
    model: YOLO,
    quantize: int | None,
    data: str | None,
    imgsz: int,
) -> Path:
    kwargs: dict[str, object] = {
        "format": "litert",
        "imgsz": imgsz,
        "nms": False,
    }

    if quantize is not None:
        kwargs.update(
            {
                "quantize": quantize,
                "data": data,
            }
        )

    try:
        return Path(
            model.export(
                **kwargs
            )
        )

    except (TypeError, ValueError):
        legacy = {
            "format": "tflite",
            "imgsz": imgsz,
            "nms": False,
        }

        if quantize is not None:
            legacy.update(
                {
                    "int8": True,
                    "data": data,
                }
            )

        return Path(
            model.export(
                **legacy
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Exporta FP32 e INT8 com calibracao do dominio."
        )
    )

    parser.add_argument(
        "--model",
        required=True,
    )

    parser.add_argument(
        "--calibration-data",
        default="configs/calibration.yaml",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
    )

    parser.add_argument(
        "--output",
        default="artifacts/exports",
    )

    args = parser.parse_args()

    source_model = resolve_path(
        args.model
    )

    output_dir = resolve_path(
        args.output
    )

    model = YOLO(
        str(source_model)
    )

    fp32_raw = export_litert(
        model,
        quantize=None,
        data=None,
        imgsz=args.imgsz,
    )

    fp32 = copy_export(
        fp32_raw,
        output_dir
        / (
            f"{source_model.stem}_"
            f"{args.imgsz}_fp32.tflite"
        ),
    )

    int8_raw = export_litert(
        model,
        quantize=8,
        data=str(
            resolve_path(
                args.calibration_data
            )
        ),
        imgsz=args.imgsz,
    )

    int8 = copy_export(
        int8_raw,
        output_dir
        / (
            f"{source_model.stem}_"
            f"{args.imgsz}_int8.tflite"
        ),
    )

    payload = {
        "source_model": str(
            source_model
        ),
        "source_sha256": sha256_file(
            source_model
        ),
        "calibration_yaml": str(
            resolve_path(
                args.calibration_data
            )
        ),
        "image_size": args.imgsz,
        "exports": {
            "fp32": {
                "path": str(fp32),
                "bytes": fp32.stat().st_size,
                "sha256": sha256_file(
                    fp32
                ),
            },
            "int8": {
                "path": str(int8),
                "bytes": int8.stat().st_size,
                "sha256": sha256_file(
                    int8
                ),
            },
        },
        "required_next_step": (
            "Avaliar ambos no mesmo teste "
            "e no mesmo aparelho."
        ),
    }

    write_json(
        output_dir
        / (
            f"{source_model.stem}_"
            f"{args.imgsz}_export_manifest.json"
        ),
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
