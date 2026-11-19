"""Сборка таблиц MITRE из JSONL с гибкими путями ввода/вывода.

Примеры запуска:
    # Три файла, вывод в папку mitre_tables/
    python build_mitre_tables.py train.jsonl test.jsonl validation.jsonl -o mitre_tables/

    # Все .jsonl из папки data/, вывод в out/
    python build_mitre_tables.py data/ -o out/

    # Один файл, автоопределение сплита по имени
    python build_mitre_tables.py data/train.jsonl -o out/

    # Явно задать сплиты через префиксы (train:, test:, val:)
    python build_mitre_tables.py train:data/train.jsonl test:data/test.jsonl val:data/val.jsonl -o out/
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


TEXT_KEYS = {"event_text", "technique_description", "technique_name"}
PREFIX = {
    "event_text": "eve", "technique_description": "tec", "technique_name": "tna",
    "label": "lab", "parent_label": "par", "tactics": "tac", "event_text_id": "eid",
}


# =============================================================
# ID-ГЕНЕРАТОРЫ
# =============================================================

def make_uid(value: str, key: str) -> str:
    h = hashlib.md5(value.encode("utf-8")).hexdigest()[:10]
    return f"{PREFIX.get(key, 'val')}_{h}"


def make_object_uid(row: dict) -> str:
    """Только от содержимого — стабильно и без привязки к файлу."""
    content = json.dumps(row, sort_keys=True, ensure_ascii=False)
    return "obj_" + hashlib.md5(content.encode("utf-8")).hexdigest()[:12]


# =============================================================
# ВВОД
# =============================================================

def collect_inputs(items: list[str]) -> list[tuple[str, Path]]:
    """
    Разворачивает аргументы в список (split, path).
    Поддерживает:
      - явный префикс: train:data/train.jsonl
      - автоопределение по имени: train.jsonl → train
      - папку: берёт все .jsonl, сплит — по имени файла
    """
    result: list[tuple[str, Path]] = []

    for item in items:
        # Явный split: "train:path"
        if ":" in item and not item.startswith(("/", "\\")) and not (len(item) > 1 and item[1] == ":"):
            split, path_str = item.split(":", 1)
            path = Path(path_str)
            if path.is_file():
                result.append((split, path))
            else:
                print(f"[!] Не найден: {path_str}", file=sys.stderr)
            continue

        path = Path(item)

        if path.is_dir():
            for f in sorted(path.glob("*.jsonl")):
                split = guess_split(f.name)
                result.append((split, f))
        elif path.is_file():
            split = guess_split(path.name)
            result.append((split, path))
        else:
            print(f"[!] Не найден: {item}", file=sys.stderr)

    return result


def guess_split(filename: str) -> str:
    """Угадывает сплит по имени файла."""
    name = filename.lower()
    if "train" in name:
        return "train"
    if "test" in name:
        return "test"
    if "val" in name or "dev" in name:
        return "validation"
    return "unknown"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


# =============================================================
# СБОРКА
# =============================================================

def build_tables(inputs: list[tuple[str, Path]]):
    params: dict[str, dict] = {}
    objects: dict[str, dict] = {}
    object_splits: list[dict] = []
    object_params: list[dict] = []
    texts: dict[str, dict] = {}
    seen_obj_params: set[tuple[str, str]] = set()
    seen_obj_splits: set[tuple[str, str]] = set()

    for split, path in inputs:
        print(f"{split}: {path.name}")
        rows = load_jsonl(path)
        print(f"  строк: {len(rows)}")

        for row in rows:
            obj_uid = make_object_uid(row)

            if obj_uid not in objects:
                objects[obj_uid] = {"object_uid": obj_uid, "source": path.stem}

            os_key = (obj_uid, split)
            if os_key not in seen_obj_splits:
                seen_obj_splits.add(os_key)
                object_splits.append({
                    "object_uid": obj_uid,
                    "split": split,
                    "file_name": path.name,
                })

            for key, value in row.items():
                if value is None:
                    continue
                if isinstance(value, (list, dict)):
                    value = json.dumps(value, ensure_ascii=False)
                value_str = str(value).strip()
                if not value_str:
                    continue

                uid = make_uid(value_str, key)
                params[uid] = {"uid": uid, "key": key, "value": value_str}

                if key in TEXT_KEYS and uid not in texts:
                    texts[uid] = {
                        "value_uid": uid,
                        "text_ru": "",
                        "text_en": value_str,
                        "correct": 0,
                        "source_key": key,
                    }

                op = (obj_uid, uid)
                if op not in seen_obj_params:
                    seen_obj_params.add(op)
                    object_params.append({"object_uid": obj_uid, "uid": uid, "key": key})

    return (
        pd.DataFrame(params.values()),
        pd.DataFrame(objects.values()),
        pd.DataFrame(object_splits),
        pd.DataFrame(object_params),
        pd.DataFrame(texts.values()),
    )


# =============================================================
# ПРОВЕРКИ
# =============================================================

def check_integrity(df_params, df_objects, df_object_splits, df_obj_params, df_texts) -> dict:
    checks = {}
    op_uids = set(df_obj_params["uid"])
    param_uids = set(df_params["uid"])
    checks["object_params.uid_not_in_params"] = len(op_uids - param_uids)

    op_objs = set(df_obj_params["object_uid"])
    obj_uids = set(df_objects["object_uid"])
    checks["object_params.object_uid_not_in_objects"] = len(op_objs - obj_uids)

    os_objs = set(df_object_splits["object_uid"])
    checks["object_splits.object_uid_not_in_objects"] = len(os_objs - obj_uids)

    text_uids = set(df_params[df_params["key"].isin(TEXT_KEYS)]["uid"])
    checks["text_uids_missing_in_texts"] = len(text_uids - set(df_texts["value_uid"]))

    checks["params_duplicate_uid"] = int(df_params["uid"].duplicated().sum())
    checks["objects_duplicate_uid"] = int(df_objects["object_uid"].duplicated().sum())
    checks["object_params_duplicates"] = int(
        df_obj_params.duplicated(subset=["object_uid", "uid"]).sum()
    )
    checks["object_splits_duplicates"] = int(
        df_object_splits.duplicated(subset=["object_uid", "split"]).sum()
    )

    leak = df_object_splits.groupby("object_uid")["split"].nunique()
    checks["objects_in_multiple_splits"] = int((leak > 1).sum())
    checks["splits_present"] = sorted(df_object_splits["split"].unique().tolist())

    return checks


# =============================================================
# ОТЧЁТ
# =============================================================

def build_report(df_params, df_objects, df_object_splits, df_obj_params, df_texts, checks) -> str:
    lines = ["# Отчёт по сборке MITRE-таблиц\n"]

    lines.append("## Сводка\n")
    lines.append("| Таблица | Строк |")
    lines.append("|---|---:|")
    for name, df in [
        ("params", df_params), ("objects", df_objects),
        ("object_splits", df_object_splits),
        ("object_params", df_obj_params), ("texts", df_texts),
    ]:
        lines.append(f"| `{name}` | {len(df)} |")
    lines.append("")

    lines.append("## Строк по сплитам\n")
    lines.append("| Сплит | Объектов |")
    lines.append("|---|---:|")
    for split, count in df_object_splits["split"].value_counts().items():
        lines.append(f"| {split} | {count} |")
    lines.append("")

    lines.append("## Распределение `key` в `params`\n")
    lines.append("| key | Уникальных uid |")
    lines.append("|---|---:|")
    for k, v in df_params["key"].value_counts().items():
        lines.append(f"| `{k}` | {v} |")
    lines.append("")

    lines.append("## Тексты\n")
    lines.append("| source_key | Всего | text_ru пусто | correct=1 |")
    lines.append("|---|---:|---:|---:|")
    for sk, group in df_texts.groupby("source_key"):
        empty_ru = (group["text_ru"] == "").sum()
        correct = (group["correct"] == 1).sum()
        lines.append(f"| `{sk}` | {len(group)} | {empty_ru} | {correct} |")
    lines.append("")

    lines.append("## Проверки целостности\n")
    lines.append("| Проверка | Значение | OK? |")
    lines.append("|---|---:|---|")
    for k, v in checks.items():
        if isinstance(v, list):
            lines.append(f"| `{k}` | `{v}` | — |")
        else:
            ok = "✅" if v == 0 else "❌"
            lines.append(f"| `{k}` | {v} | {ok} |")
    lines.append("")

    lines.append("## Примеры\n")
    for name, df in [
        ("params", df_params), ("objects", df_objects),
        ("object_splits", df_object_splits),
        ("object_params", df_obj_params), ("texts", df_texts),
    ]:
        lines.append(f"### `{name}`\n")
        lines.append("```")
        lines.append(df.head(5).to_string(index=False))
        lines.append("```\n")

    return "\n".join(lines)


# =============================================================
# MAIN
# =============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Сборка таблиц MITRE из JSONL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("inputs", nargs="+",
                        help="Файлы .jsonl, папки или 'split:path'")
    parser.add_argument("-o", "--output", default="mitre_tables",
                        help="Папка для Parquet и отчёта (по умолчанию: mitre_tables)")
    args = parser.parse_args()

    inputs = collect_inputs(args.inputs)
    if not inputs:
        print("[!] Нет входных файлов.", file=sys.stderr)
        sys.exit(1)

    print(f"\n=== Входные файлы ({len(inputs)}) ===")
    for split, path in inputs:
        print(f"  {split}: {path}")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    print("\n=== Сборка ===")
    df_params, df_objects, df_object_splits, df_obj_params, df_texts = build_tables(inputs)

    print("\n=== Проверки целостности ===")
    checks = check_integrity(df_params, df_objects, df_object_splits, df_obj_params, df_texts)
    for k, v in checks.items():
        print(f"  {k}: {v}")

    print("\n=== Сохранение ===")
    outputs = {
        "params": df_params,
        "objects": df_objects,
        "object_splits": df_object_splits,
        "object_params": df_obj_params,
        "texts": df_texts,
    }
    for name, df in outputs.items():
        path = output / f"{name}.parquet"
        df.to_parquet(path, index=False)
        print(f"  → {path}")

    report = build_report(df_params, df_objects, df_object_splits, df_obj_params, df_texts, checks)
    report_path = output / "report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"  → {report_path}")


if __name__ == "__main__":
    main()