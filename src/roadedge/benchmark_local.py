from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from roadedge.common import (
    resolve_path,
    sha256_file,
    write_json,
)


def percentile(
    values: list[float],
    q: float,
) -> float:
    return float(
        np.percentile(
            np.asarray(
                values,
                dtype=float,
            ),
            q,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Microbenchmark local; "
            "nao substitui o teste Android."
        )
    )

    parser.add_argument(
        "--model",
        required=True,
    )

    parser.add_argument(
        "--images",
        default="data/splits/test.txt",
        help="TXT com caminhos das imagens de teste.",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--device",
        default="cpu",
    )

    parser.add_argument(
        "--output",
        default="artifacts/benchmarks",
    )

    args = parser.parse_args()

    model_path = resolve_path(
        args.model
    )

    paths = [
        Path(line)
        for line
        in resolve_path(
            args.images
        ).read_text().splitlines()
        if line.strip()
    ]

    if not paths:
        raise ValueError(
            "A lista de imagens esta vazia."
        )

    model = YOLO(
        str(model_path)
    )

    for index in range(
        args.warmup
    ):
        model.predict(
            str(
                paths[
                    index % len(paths)
                ]
            ),
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )

    samples_ms: list[float] = []

    for index in range(
        args.runs
    ):
        started = (
            time.perf_counter_ns()
        )

        model.predict(
            str(
                paths[
                    index % len(paths)
                ]
            ),
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )

        samples_ms.append(
            (
                time.perf_counter_ns()
                - started
            )
            / 1_000_000.0
        )

    payload = {
        "model": str(
            model_path
        ),
        "model_sha256": sha256_file(
            model_path
        ),
        "model_bytes": (
            model_path.stat().st_size
        ),
        "device": args.device,
        "image_size": args.imgsz,
        "warmup_runs": args.warmup,
        "measured_runs": args.runs,
        "latency_ms": {
            "mean": (
                statistics.fmean(
                    samples_ms
                )
            ),
            "median": (
                statistics.median(
                    samples_ms
                )
            ),
            "p90": percentile(
                samples_ms,
                90,
            ),
            "p95": percentile(
                samples_ms,
                95,
            ),
            "max": max(
                samples_ms
            ),
        },
        "throughput_fps_from_median": (
            1000.0
            / statistics.median(
                samples_ms
            )
        ),
        "warning": (
            "Inclui pre e pos-processamento Python; "
            "reporte separadamente do benchmark no celular."
        ),
    }

    output = (
        resolve_path(
            args.output
        )
        / (
            f"{model_path.stem}_"
            f"{args.device}.json"
        )
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
