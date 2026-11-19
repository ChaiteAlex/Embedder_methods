"""Собирает датасет "описание -> описание" из обогащённых JSONL.

Вход  : intermediate/{train,validation,test}.jsonl  (после merge_dataset.py)
Выход : final/{split}.jsonl
        final/{split}_unlabeled.jsonl

Схема строки (labeled):
    {
      "event_text_id": "<sha1(event_text)[:12]>",
      "event_text": "...",
      "technique_name": "...",
      "technique_description": "...",
      "label": "T1505.003",
      "parent_label": "T1505",
      "tactics": "Persistence"
    }

Multi-label: один event_text с N метками -> N строк с одинаковым event_text_id.
Пустые техники (data == null) пропускаются: без description пара бессмысленна.
"""

import argparse
import hashlib
import json
from pathlib import Path

# 1. Определяем корень проекта (скрипт лежит в scripts/)
HERE = Path(__file__).resolve().parent          # .../project_root/scripts
ROOT_DIR = HERE.parent                          # .../project_root

# 2. Настраиваем пути по умолчанию согласно структуре проекта
INPUT_DIR = ROOT_DIR / "raw"
OUTPUT_DIR = ROOT_DIR / "final" / "rebuild_mitra"

SPLITS = ("train", "validation", "test")


def short_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


def get_event_text(obj: dict) -> str:
    # В обогащённом JSONL исходный столбец называется text1.
    # Оставляем fallback на event_text на случай переименования.
    for key in ("text1", "event_text"):
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def explode_labeled(obj: dict) -> list[dict]:
    text = get_event_text(obj)
    if not text:
        return []
    text_id = short_hash(text)
    rows = []
    for detail in obj.get("attack_details") or []:
        label = detail.get("label")
        data = detail.get("data") or {}
        description = (data.get("description") or "").strip()
        if not label or not description:
            continue
        
        rows.append({
            "event_text": text,
            "technique_name": (data.get("name") or "").strip(),
            "technique_description": description,
            "label": label,
            "parent_label": data.get("sub-technique of") or label,
            "tactics": (data.get("tactics") or "").strip(),
        })
    return rows


def build_unlabeled(obj: dict) -> list[dict]:
    text = get_event_text(obj)
    if not text:
        return []
    text_id = short_hash(text)
    return [{
        "event_text_id": text_id,
        "event_text": text,
        "technique_name": "",
        "technique_description": "",
        "label": None,
        "parent_label": None,
        "tactics": "",
    }]


def write_split(split: str, src: Path, labeled: Path) -> dict:
    stats = {"rows_in": 0, "labeled": 0, "unlabeled": 0,
             "skipped_empty_text": 0, "skipped_missing_data": 0,
             "multi_label_events": 0, "unique_labels": set(), "unique_parents": set()}
    seen_texts_with_multi = set()

    # Убедимся, что папка для вывода существует
    labeled.parent.mkdir(parents=True, exist_ok=True)

    with src.open(encoding="utf-8-sig") as fin, \
         labeled.open("w", encoding="utf-8", newline="\n") as f_lab:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            stats["rows_in"] += 1

            if not get_event_text(obj):
                stats["skipped_empty_text"] += 1
                continue

            lab_rows = explode_labeled(obj)
            missing = len(obj.get("attack_details") or []) - len(lab_rows)
            if missing > 0:
                stats["skipped_missing_data"] += missing

            if len(lab_rows) > 1:
                stats["multi_label_events"] += 1

            for row in lab_rows:
                f_lab.write(json.dumps(row, ensure_ascii=False) + "\n")
                stats["labeled"] += 1
                stats["unique_labels"].add(row["label"])
                stats["unique_parents"].add(row["parent_label"])


    stats["unique_labels"] = len(stats["unique_labels"])
    stats["unique_parents"] = len(stats["unique_parents"])
    stats["multi_label_unique_texts"] = len(seen_texts_with_multi)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--root-dir", type=Path, default=ROOT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {}

    for split in SPLITS:
        src = args.input_dir / f"{split}.jsonl"
        if not src.exists():
            print(f"[skip] {src} не найден")
            continue
        lab = args.output_dir / f"{split}.jsonl"
        
        stats = write_split(split, src, lab)
        report[split] = stats
        print(f"{split}: in={stats['rows_in']} labeled={stats['labeled']} "
              f"unlabeled={0} multi_label_events={stats['multi_label_events']} "
              f"unique_labels={stats['unique_labels']}")

    (args.root_dir / "report" / "clean_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nОтчёт: {args.output_dir / 'clean_report.json'}")


if __name__ == "__main__":
    main()