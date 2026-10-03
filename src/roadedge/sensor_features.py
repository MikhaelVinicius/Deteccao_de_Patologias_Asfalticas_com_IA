from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from roadedge.common import resolve_path


SIGNALS = (
    "ax",
    "ay",
    "az",
    "magnitude",
)


def summarize(
    values: np.ndarray,
    prefix: str,
) -> dict[str, float]:
    return {
        f"{prefix}_mean": float(
            np.mean(values)
        ),
        f"{prefix}_std": float(
            np.std(values)
        ),
        f"{prefix}_rms": float(
            np.sqrt(
                np.mean(
                    np.square(values)
                )
            )
        ),
        f"{prefix}_min": float(
            np.min(values)
        ),
        f"{prefix}_max": float(
            np.max(values)
        ),
        f"{prefix}_p2p": float(
            np.ptp(values)
        ),
        f"{prefix}_median": float(
            np.median(values)
        ),
        f"{prefix}_iqr": float(
            np.percentile(values, 75)
            - np.percentile(values, 25)
        ),
        f"{prefix}_absmax": float(
            np.max(
                np.abs(values)
            )
        ),
    }


def make_windows(
    frame: pd.DataFrame,
    window_samples: int,
    step_samples: int,
) -> pd.DataFrame:
    required = {
        "timestamp_ns",
        "ax",
        "ay",
        "az",
        "speed_mps",
        "label",
        "trip_id",
    }

    missing = (
        required
        - set(frame.columns)
    )

    if missing:
        raise ValueError(
            f"Colunas ausentes: {sorted(missing)}"
        )

    ordered = frame.sort_values(
        [
            "trip_id",
            "timestamp_ns",
        ]
    ).copy()

    ordered["magnitude"] = np.sqrt(
        ordered["ax"] ** 2
        + ordered["ay"] ** 2
        + ordered["az"] ** 2
    )

    rows: list[
        dict[
            str,
            float | str | int,
        ]
    ] = []

    for trip_id, trip in ordered.groupby(
        "trip_id",
        sort=False,
    ):
        trip = trip.reset_index(
            drop=True
        )

        for start in range(
            0,
            len(trip)
            - window_samples
            + 1,
            step_samples,
        ):
            window = trip.iloc[
                start:
                start + window_samples
            ]

            features: dict[
                str,
                float | str | int,
            ] = {
                "trip_id": str(
                    trip_id
                ),
                "window_start_ns": int(
                    window[
                        "timestamp_ns"
                    ].iloc[0]
                ),
                "window_end_ns": int(
                    window[
                        "timestamp_ns"
                    ].iloc[-1]
                ),
                "speed_mean_mps": float(
                    window[
                        "speed_mps"
                    ].mean()
                ),
                "label": int(
                    window[
                        "label"
                    ].mode().iloc[0]
                ),
            }

            for signal in SIGNALS:
                features.update(
                    summarize(
                        window[
                            signal
                        ].to_numpy(
                            dtype=float
                        ),
                        signal,
                    )
                )

            rows.append(
                features
            )

    return pd.DataFrame(
        rows
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extrai atributos moveis de aceleracao."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--window-seconds",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--step-seconds",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--output",
        default=(
            "data/processed/"
            "sensor_windows.csv"
        ),
    )

    args = parser.parse_args()

    raw = pd.read_csv(
        resolve_path(
            args.input
        )
    )

    windows = make_windows(
        raw,
        window_samples=round(
            args.sample_rate
            * args.window_seconds
        ),
        step_samples=round(
            args.sample_rate
            * args.step_seconds
        ),
    )

    target = resolve_path(
        args.output
    )

    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    windows.to_csv(
        target,
        index=False,
    )

    print(
        f"{len(windows)} janelas "
        f"salvas em {target}"
    )


if __name__ == "__main__":
    main()
