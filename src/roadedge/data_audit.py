from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import imagehash
from PIL import Image, UnidentifiedImageError

from roadedge.common import resolve_path, sha256_file, write_json


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def parse_label(label_path: Path, class_count: int) -> tuple[list[int], list[str]]:
    classes: list[int] = []
    problems: list[str] = []

    if not label_path.exists():
        return classes, ["rotulo_ausente"]

    for line_number, raw in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not raw.strip():
            continue

        parts = raw.split()

        if len(parts) != 5:
            problems.append(f"linha_{line_number}:esperado_5_campos")
            continue

        try:
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
        except ValueError:
            problems.append(f"linha_{line_number}:valor_invalido")
            continue

        if not 0 <= class_id < class_count:
            problems.append(f"linha_{line_number}:classe_fora_do_intervalo")

        if not all(
            0.0 <= value <= 1.0
            for value in (x_center, y_center, width, height)
        ):
            problems.append(f"linha_{line_number}:coordenada_fora_de_0_1")

        if width <= 0.0 or height <= 0.0:
            problems.append(f"linha_{line_number}:caixa_sem_area")

        if width * height > 0.95:
            problems.append(f"linha_{line_number}:caixa_suspeita_maior_95pct")

        classes.append(class_id)

    return classes, problems


def audit(
    images_dir: Path,
    labels_dir: Path,
    class_count: int,
) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    exact_hashes: dict[str, list[str]] = defaultdict(list)
    perceptual_hashes: dict[str, list[str]] = defaultdict(list)
    class_instances: Counter[int] = Counter()
    problem_counts: Counter[str] = Counter()

    images = sorted(
        path
        for path in images_dir.rglob("*")
        if path.suffix.lower() in IMAGE_SUFFIXES
    )

    for image_path in images:
        relative = image_path.relative_to(images_dir)
        label_path = labels_dir / relative.with_suffix(".txt")

        problems: list[str] = []
        width = height = 0
        sha256 = ""
        phash = ""

        try:
            with Image.open(image_path) as image:
                image.verify()

            with Image.open(image_path) as image:
                width, height = image.size
                phash = str(imagehash.phash(image.convert("RGB")))

            sha256 = sha256_file(image_path)

        except (OSError, UnidentifiedImageError):
            problems.append("imagem_corrompida")

        classes, label_problems = parse_label(label_path, class_count)
        problems.extend(label_problems)

        class_instances.update(classes)

        for problem in problems:
            problem_counts[problem.split(":")[-1]] += 1

        if sha256:
            exact_hashes[sha256].append(str(image_path))

        if phash:
            perceptual_hashes[phash].append(str(image_path))

        rows.append(
            {
                "image": str(image_path.resolve()),
                "label": str(label_path.resolve()),
                "width": width,
                "height": height,
                "class_ids": ";".join(map(str, sorted(set(classes)))),
                "instances": len(classes),
                "sha256": sha256,
                "phash": phash,
                "problems": ";".join(problems),
            }
        )

    duplicate_sets = [
        paths for paths in exact_hashes.values() if len(paths) > 1
    ]

    visual_candidate_sets = [
        paths for paths in perceptual_hashes.values() if len(paths) > 1
    ]

    summary = {
        "images": len(images),
        "labeled_images": sum(
            Path(row["label"]).exists() for row in rows
        ),
        "background_images": sum(
            row["instances"] == 0 and not row["problems"]
            for row in rows
        ),
        "class_instances": dict(sorted(class_instances.items())),
        "problem_counts": dict(problem_counts),
        "exact_duplicate_sets": duplicate_sets,
        "same_phash_candidate_sets": visual_candidate_sets,
        "warning": (
            "Revise manualmente candidatos de pHash; "
            "igualdade de hash nao prova duplicacao."
        ),
    }

    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audita imagens e rotulos no formato YOLO."
    )
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--classes", type=int, required=True)
    parser.add_argument("--output", default="artifacts/data_audit")
    args = parser.parse_args()

    output_dir = resolve_path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, summary = audit(
        resolve_path(args.images),
        resolve_path(args.labels),
        args.classes,
    )

    with (output_dir / "manifest_audit.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=rows[0].keys() if rows else ["image"],
        )
        writer.writeheader()
        writer.writerows(rows)

    write_json(output_dir / "summary.json", summary)

    print(f"Auditoria salva em {output_dir}")
    print(
        f"Imagens: {summary['images']} | "
        f"problemas: {sum(summary['problem_counts'].values())}"
    )


if __name__ == "__main__":
    main()
