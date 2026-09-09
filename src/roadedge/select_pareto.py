from __future__ import annotations

import argparse

import pandas as pd

from roadedge.common import resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Filtra candidatos Edge por restricoes pre-declaradas."
        )
    )

    parser.add_argument(
        "--results",
        required=True,
    )

    parser.add_argument(
        "--max-p95-ms",
        type=float,
        default=100.0,
    )

    parser.add_argument(
        "--max-size-mb",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--max-map-drop-pp",
        type=float,
        default=3.0,
    )

    parser.add_argument(
        "--output",
        default="artifacts/model_selection.csv",
    )

    args = parser.parse_args()

    frame = pd.read_csv(
        resolve_path(
            args.results
        )
    )

    required = {
        "candidate",
        "map50_95",
        "p95_ms",
        "size_mb",
        "map_drop_pp",
    }

    missing = (
        required
        - set(
            frame.columns
        )
    )

    if missing:
        raise ValueError(
            f"Colunas ausentes: {sorted(missing)}"
        )

    frame["eligible"] = (
        (
            frame["p95_ms"]
            <= args.max_p95_ms
        )
        & (
            frame["size_mb"]
            <= args.max_size_mb
        )
        & (
            frame["map_drop_pp"]
            <= args.max_map_drop_pp
        )
    )

    frame["pareto"] = False

    eligible = frame[
        frame["eligible"]
    ]

    for index, row in eligible.iterrows():
        dominated = eligible[
            (
                eligible["map50_95"]
                >= row["map50_95"]
            )
            & (
                eligible["p95_ms"]
                <= row["p95_ms"]
            )
            & (
                eligible["size_mb"]
                <= row["size_mb"]
            )
            & (
                (
                    eligible["map50_95"]
                    > row["map50_95"]
                )
                | (
                    eligible["p95_ms"]
                    < row["p95_ms"]
                )
                | (
                    eligible["size_mb"]
                    < row["size_mb"]
                )
            )
        ]

        frame.loc[
            index,
            "pareto",
        ] = dominated.empty

    frame = frame.sort_values(
        [
            "eligible",
            "pareto",
            "map50_95",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    )

    target = resolve_path(
        args.output
    )

    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_csv(
        target,
        index=False,
    )

    print(
        frame.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
