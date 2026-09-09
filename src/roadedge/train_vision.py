from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from roadedge.common import (
    git_commit,
    load_yaml,
    resolve_path,
    set_seed,
    write_json,
)


def plan(
    config: dict[str, Any],
    phase: str,
) -> list[dict[str, Any]]:
    if phase == "screening":
        return [
            {
                "model": model,
                "imgsz": int(imgsz),
                "seed": int(config["screening_seed"]),
                "epochs": int(config["epochs_screening"]),
            }
            for model in config["models"]
            for imgsz in config["image_sizes"]
        ]

    finalists = config.get("finalists", [])

    if not finalists:
        raise ValueError(
            "Preencha finalists no YAML usando a "
            "fronteira de Pareto da triagem."
        )

    return [
        {
            "model": candidate["model"],
            "imgsz": int(candidate["imgsz"]),
            "seed": int(seed),
            "epochs": int(config["epochs_final"]),
        }
        for candidate in finalists
        for seed in config["final_seeds"]
    ]


def append_summary(record: dict[str, Any]) -> None:
    target = resolve_path(
        "artifacts/reports/train_matrix.csv"
    )

    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    exists = target.exists()

    columns = [
        "run_name",
        "phase",
        "base_model",
        "image_size",
        "seed",
        "elapsed_seconds",
        "map50",
        "map50_95",
        "best_weights",
        "git_commit",
    ]

    with target.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=columns,
        )

        if not exists:
            writer.writeheader()

        writer.writerow(
            {
                key: record.get(key, "")
                for key in columns
            }
        )


def run_one(
    config: dict[str, Any],
    phase: str,
    item: dict[str, Any],
) -> dict[str, Any]:
    set_seed(item["seed"])

    model_stem = Path(
        item["model"]
    ).stem

    run_name = (
        f"{phase}_"
        f"{model_stem}_"
        f"{item['imgsz']}_"
        f"seed{item['seed']}"
    )

    started = time.perf_counter()

    result = YOLO(
        item["model"]
    ).train(
        data=str(
            resolve_path(
                config["data"]
            )
        ),
        epochs=item["epochs"],
        imgsz=item["imgsz"],
        batch=config["batch"],
        patience=config["patience"],
        device=config["device"],
        workers=config["workers"],
        optimizer=config["optimizer"],
        lr0=config["lr0"],
        weight_decay=config["weight_decay"],
        cos_lr=config["cos_lr"],
        close_mosaic=config["close_mosaic"],
        deterministic=config["deterministic"],
        seed=item["seed"],
        pretrained=True,
        project=str(
            resolve_path(
                "artifacts/vision_runs"
            )
        ),
        name=run_name,
        exist_ok=False,
        plots=True,
        verbose=True,
    )

    metrics = {
        key: float(value)
        for key, value
        in result.results_dict.items()
    }

    save_dir = Path(
        result.save_dir
    )

    record = {
        "run_name": run_name,
        "phase": phase,
        "base_model": item["model"],
        "transfer_learning": True,
        "image_size": item["imgsz"],
        "seed": item["seed"],
        "epochs_requested": item["epochs"],
        "elapsed_seconds": (
            time.perf_counter() - started
        ),
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "metrics_validation": metrics,
        "map50": metrics.get(
            "metrics/mAP50(B)"
        ),
        "map50_95": metrics.get(
            "metrics/mAP50-95(B)"
        ),
        "best_weights": str(
            (
                save_dir
                / "weights"
                / "best.pt"
            ).resolve()
        ),
        "config": config,
    }

    write_json(
        save_dir / "run_manifest.json",
        record,
    )

    append_summary(record)

    return record


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Executa matriz visual rastreavel."
        )
    )

    parser.add_argument(
        "--config",
        default="configs/experiment.yaml",
    )

    parser.add_argument(
        "--phase",
        choices=[
            "screening",
            "final",
        ],
        required=True,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = parser.parse_args()

    config = load_yaml(
        args.config
    )

    execution_plan = plan(
        config,
        args.phase,
    )

    print(
        json.dumps(
            execution_plan,
            indent=2,
            ensure_ascii=False,
        )
    )

    if args.dry_run:
        return

    for item in execution_plan:
        record = run_one(
            config,
            args.phase,
            item,
        )

        print(
            json.dumps(
                record,
                indent=2,
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
