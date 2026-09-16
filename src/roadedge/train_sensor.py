from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    GroupShuffleSplit,
    RandomizedSearchCV,
    StratifiedGroupKFold,
)

from roadedge.common import (
    resolve_path,
    write_json,
)
from roadedge.forest_json import (
    export_forest,
    load_payload,
    predict_proba_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Treina RF sem misturar janelas "
            "da mesma viagem."
        )
    )

    parser.add_argument(
        "--features",
        default="data/processed/sensor_windows.csv",
    )

    parser.add_argument(
        "--output",
        default="artifacts/sensor",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    frame = pd.read_csv(
        resolve_path(
            args.features
        )
    )

    metadata = {
        "trip_id",
        "window_start_ns",
        "window_end_ns",
        "label",
    }

    feature_names = [
        column
        for column in frame.columns
        if column not in metadata
    ]

    x = frame[
        feature_names
    ]

    y = frame[
        "label"
    ].astype(int)

    groups = frame[
        "trip_id"
    ].astype(str)

    holdout = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=args.seed,
    )

    train_idx, test_idx = next(
        holdout.split(
            x,
            y,
            groups,
        )
    )

    x_train = x.iloc[
        train_idx
    ]

    x_test = x.iloc[
        test_idx
    ]

    y_train = y.iloc[
        train_idx
    ]

    y_test = y.iloc[
        test_idx
    ]

    groups_train = groups.iloc[
        train_idx
    ]

    estimator = RandomForestClassifier(
        class_weight="balanced_subsample",
        random_state=args.seed,
        n_jobs=-1,
    )

    search = RandomizedSearchCV(
        estimator,
        param_distributions={
            "n_estimators": [
                100,
                200,
                300,
            ],
            "max_depth": [
                6,
                10,
                14,
                None,
            ],
            "min_samples_leaf": [
                1,
                2,
                4,
                8,
            ],
            "max_features": [
                "sqrt",
                0.5,
                0.8,
            ],
        },
        n_iter=20,
        scoring="f1_macro",
        cv=StratifiedGroupKFold(
            n_splits=5,
            shuffle=True,
            random_state=args.seed,
        ),
        n_jobs=-1,
        random_state=args.seed,
        refit=True,
        verbose=1,
    )

    search.fit(
        x_train,
        y_train,
        groups=groups_train,
    )

    prediction = (
        search.best_estimator_.predict(
            x_test
        )
    )

    output_dir = resolve_path(
        args.output
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        search.best_estimator_,
        output_dir
        / "random_forest.joblib",
    )

    export_forest(
        search.best_estimator_,
        feature_names,
        output_dir
        / "random_forest.json",
    )

    payload = load_payload(
        output_dir
        / "random_forest.json"
    )

    json_prediction = [
        payload["classes"][
            int(
                predict_proba_json(
                    payload,
                    row.to_dict(),
                ).argmax()
            )
        ]
        for _, row
        in x_test.iterrows()
    ]

    parity = float(
        (
            pd.Series(
                json_prediction,
                index=y_test.index,
            )
            == prediction
        ).mean()
    )

    report = {
        "best_params": (
            search.best_params_
        ),
        "cv_best_f1_macro": float(
            search.best_score_
        ),
        "test_f1_macro": float(
            f1_score(
                y_test,
                prediction,
                average="macro",
            )
        ),
        "test_confusion_matrix": (
            confusion_matrix(
                y_test,
                prediction,
            ).tolist()
        ),
        "test_report": (
            classification_report(
                y_test,
                prediction,
                output_dict=True,
                zero_division=0,
            )
        ),
        "train_trips": sorted(
            groups.iloc[
                train_idx
            ].unique().tolist()
        ),
        "test_trips": sorted(
            groups.iloc[
                test_idx
            ].unique().tolist()
        ),
        "json_export_parity": parity,
        "acceptance": (
            "A paridade do JSON deve ser "
            "1.0 antes do uso movel."
        ),
    }

    write_json(
        output_dir
        / "sensor_metrics.json",
        report,
    )

    print(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
