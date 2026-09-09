from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from roadedge.common import resolve_path, write_json


def class_vector(series: pd.Series, class_count: int) -> np.ndarray:
    vector = np.zeros(class_count, dtype=float)

    for value in series.fillna(""):
        for token in str(value).split(";"):
            if token.strip():
                vector[int(token)] += 1.0

    return vector


def split_score(
    parts: dict[str, pd.DataFrame],
    class_count: int,
) -> float:
    target_sizes = {
        "train": 0.70,
        "val": 0.15,
        "test": 0.15,
    }

    total_rows = sum(len(frame) for frame in parts.values())

    global_classes = class_vector(
        pd.concat(parts.values())["class_ids"],
        class_count,
    )

    global_distribution = global_classes / max(
        global_classes.sum(),
        1.0,
    )

    score = 0.0

    for name, frame in parts.items():
        score += abs(
            len(frame) / total_rows - target_sizes[name]
        )

        local = class_vector(
            frame["class_ids"],
            class_count,
        )

        local_distribution = local / max(
            local.sum(),
            1.0,
        )

        score += float(
            np.abs(
                local_distribution - global_distribution
            ).mean()
        )

        if np.any(local == 0) and np.any(global_classes > 0):
            score += 2.0

    return score


def candidate_split(
    frame: pd.DataFrame,
    seed: int,
) -> dict[str, pd.DataFrame]:
    first = GroupShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=seed,
    )

    train_idx, temp_idx = next(
        first.split(
            frame,
            groups=frame["group_id"],
        )
    )

    train = frame.iloc[train_idx].copy()
    temp = frame.iloc[temp_idx].copy()

    second = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=seed + 10_000,
    )

    val_idx, test_idx = next(
        second.split(
            temp,
            groups=temp["group_id"],
        )
    )

    return {
        "train": train,
        "val": temp.iloc[val_idx].copy(),
        "test": temp.iloc[test_idx].copy(),
    }


def assert_no_leakage(
    parts: dict[str, pd.DataFrame],
) -> None:
    groups = {
        name: set(frame["group_id"])
        for name, frame in parts.items()
    }

    assert groups["train"].isdisjoint(groups["val"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["val"].isdisjoint(groups["test"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Cria split por rota/video/dia, "
            "sem vazar grupos."
        )
    )

    parser.add_argument(
        "--manifest",
        required=True,
    )

    parser.add_argument(
        "--classes",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--output",
        default="data/splits",
    )

    args = parser.parse_args()

    manifest = pd.read_csv(
        resolve_path(args.manifest)
    )

    required = {
        "image",
        "group_id",
        "class_ids",
    }

    missing = required - set(manifest.columns)

    if missing:
        raise ValueError(
            f"Colunas ausentes no manifesto: "
            f"{sorted(missing)}"
        )

    if manifest["group_id"].nunique() < 10:
        raise ValueError(
            "Use ao menos 10 grupos independentes "
            "para um split defensavel."
        )

    best_parts: dict[str, pd.DataFrame] | None = None
    best_seed = -1
    best_score = float("inf")

    for seed in range(args.trials):
        parts = candidate_split(
            manifest,
            seed,
        )

        score = split_score(
            parts,
            args.classes,
        )

        if score < best_score:
            best_parts = parts
            best_seed = seed
            best_score = score

    assert best_parts is not None

    assert_no_leakage(best_parts)

    output_dir = resolve_path(
        args.output
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    split_frames = []

    for name, frame in best_parts.items():
        frame = frame.copy()
        frame["split"] = name
        split_frames.append(frame)

        (
            output_dir / f"{name}.txt"
        ).write_text(
            "\n".join(
                frame["image"].astype(str)
            )
            + "\n",
            encoding="utf-8",
        )

    train = best_parts["train"].copy()

    calibration_groups = sorted(
        train["group_id"].unique()
    )

    calibration_pool = train[
        train["group_id"].isin(
            calibration_groups
        )
    ]

    calibration = calibration_pool.sample(
        n=min(
            500,
            len(calibration_pool),
        ),
        random_state=42,
    ).sort_index()

    (
        output_dir / "calibration.txt"
    ).write_text(
        "\n".join(
            calibration["image"].astype(str)
        )
        + "\n",
        encoding="utf-8",
    )

    combined = pd.concat(
        split_frames
    ).sort_index()

    combined.to_csv(
        output_dir / "manifest_split.csv",
        index=False,
    )

    write_json(
        output_dir / "split_report.json",
        {
            "selection_seed": best_seed,
            "balance_score": best_score,
            "rows": {
                name: len(frame)
                for name, frame
                in best_parts.items()
            },
            "groups": {
                name: frame["group_id"].nunique()
                for name, frame
                in best_parts.items()
            },
            "calibration_images": len(
                calibration
            ),
        },
    )

    print(
        f"Split salvo em {output_dir}; "
        f"seed de selecao={best_seed}; "
        f"score={best_score:.4f}"
    )


if __name__ == "__main__":
    main()
