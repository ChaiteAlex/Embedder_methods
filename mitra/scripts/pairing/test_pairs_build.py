"""
Тестовая сборка датасета пар (ТОЛЬКО позитивные пары, без негативов).

Примеры запуска:
    # Базовый запуск
    python build_pairs_test.py -i mitre_tables/ -o pairs_test/

    # Только текст-текст пары
    python build_pairs_test.py -i mitre_tables/ -o pairs_test/ --text-only
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any

import pandas as pd


# ============================================================
# КОНСТАНТЫ
# ============================================================

# Правила генерации пар: (key_1, key_2, pair_type, both_text)
PAIR_RULES: List[Tuple[str, str, str, bool]] = [
    # Текст ↔ Текст (основные пары для контрастивного обучения)
    ("event_text", "technique_description", "eve_to_tec_desc", True),
    ("event_text", "technique_name", "eve_to_tec_name", True),
    ("technique_name", "technique_description", "tec_name_to_desc", True),

    # Текст ↔ Код (опционально)
    ("event_text", "label", "eve_to_label", False),
    ("event_text", "tactics", "eve_to_tactics", False),
    ("technique_description", "label", "tec_desc_to_label", False),
    ("technique_description", "tactics", "tec_desc_to_tactics", False),
    ("technique_name", "label", "tec_name_to_label", False),
    ("technique_name", "tactics", "tec_name_to_tactics", False),

    # Код ↔ Код (структурные связи)
    ("label", "parent_label", "label_to_parent", False),
    ("label", "tactics", "label_to_tactics", False),
]


# ============================================================
# ВВОД
# ============================================================

def load_tables(args) -> Dict[str, pd.DataFrame]:
    """Загружает parquet-таблицы."""
    if args.inputs:
        base = Path(args.inputs)
        paths = {
            "object_params": base / "object_params.parquet",
            "object_splits": base / "object_splits.parquet",
        }
        print(Path(base / "object_params.parquet"))
    else:
        paths = {
            "object_params": Path(args.object_params),
            "object_splits": Path(args.object_splits),
        }

    tables = {}
    for name, path in paths.items():
        if not path.is_file():
            print(f"[!] Не найден файл: {path}", file=sys.stderr)
            sys.exit(1)
        print(f"  → {name}: {path}  ({path.stat().st_size / 1024 / 1024:.1f} MB)")
        tables[name] = pd.read_parquet(path)

    return tables


# ============================================================
# ГЕНЕРАЦИЯ ПАР
# ============================================================

def get_pair_rules(text_only: bool) -> List[Tuple[str, str, str, bool]]:
    """Фильтрует правила."""
    if text_only:
        return [r for r in PAIR_RULES if r[3]]
    return PAIR_RULES


def generate_positive_pairs(
    df_obj_params: pd.DataFrame,
    df_object_splits: pd.DataFrame,
    pair_rules: List[Tuple[str, str, str, bool]],
) -> pd.DataFrame:
    """Генерирует только позитивные пары."""
    df = df_obj_params.merge(
        df_object_splits[["object_uid", "split"]],
        on="object_uid",
        how="inner",
    )

    all_rows: List[Dict[str, Any]] = []
    grouped = df.groupby(["object_uid", "split"])

    for (obj_uid, split), group in grouped:
        key_to_uids = group.groupby("key")["uid"].apply(list).to_dict()

        for k1, k2, pair_type, _ in pair_rules:
            list1 = key_to_uids.get(k1, [])
            list2 = key_to_uids.get(k2, [])
            if not list1 or not list2:
                continue

            for u1 in list1:
                for u2 in list2:
                    all_rows.append({
                        "value_uid_1": u1,
                        "value_uid_2": u2,
                        "pair_type": pair_type,
                        "is_positive": 1,
                        "split": split,
                    })

    return pd.DataFrame(all_rows)


# ============================================================
# ПРОВЕРКИ ЦЕЛОСТНОСТИ
# ============================================================

def check_integrity(
    df_obj_params: pd.DataFrame,
    df_object_splits: pd.DataFrame,
    pos_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Проверяет целостность."""
    checks: Dict[str, Any] = {}

    # Дубликаты
    checks["pos_duplicates"] = int(pos_df.duplicated(subset=["value_uid_1", "value_uid_2", "pair_type"]).sum())

    # Сплиты
    checks["splits_in_pairs"] = sorted(pos_df["split"].unique().tolist())

    # UID существуют в object_params
    pair_uids = set(pos_df["value_uid_1"].unique()) | set(pos_df["value_uid_2"].unique())
    params_uids = set(df_obj_params["uid"].unique())
    checks["pair_uids_missing_in_params"] = len(pair_uids - params_uids)

    return checks


# ============================================================
# ОТЧЁТ
# ============================================================

def build_report(
    pos_df: pd.DataFrame,
    checks: Dict[str, Any],
    args,
) -> str:
    """Строит markdown-отчёт."""
    lines = ["# Отчёт по тестовой сборке датасета (только позитивные пары)\n"]

    # Параметры
    lines.append("## Параметры запуска\n")
    lines.append(f"- **text_only**: `{args.text_only}`")
    lines.append("")

    # Сводка
    lines.append("## Сводка по парам\n")
    lines.append("| Сплит | Позитивных |")
    lines.append("|-|-:|")
    for split in ["train", "validation", "test"]:
        pos_n = len(pos_df[pos_df["split"] == split])
        lines.append(f"| `{split}` | {pos_n:,} |")
    lines.append("")

    # Распределение по pair_type
    lines.append("## Распределение по `pair_type`\n")
    lines.append("| pair_type | Позитивных |")
    lines.append("|-|-:|")
    for pt, group in pos_df.groupby("pair_type"):
        lines.append(f"| `{pt}` | {len(group):,} |")
    lines.append("")

    # Проверки
    lines.append("## Проверки целостности\n")
    lines.append("| Проверка | Значение | OK? |")
    lines.append("|-|-:|-|")
    for k, v in checks.items():
        if isinstance(v, list):
            lines.append(f"| `{k}` | `{v}` | — |")
        else:
            ok = "✅" if v == 0 else "❌"
            lines.append(f"| `{k}` | {v} | {ok} |")
    lines.append("")

    # Примеры
    lines.append("## Примеры пар\n")
    lines.append("```")
    lines.append(pos_df.head(10).to_string(index=False))
    lines.append("```\n")

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Тестовая сборка датасета (только позитивные пары)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-i", "--inputs", help="Папка с parquet-таблицами")
    group.add_argument("--object-params", help="Путь к object_params.parquet")

    parser.add_argument("--object-splits", help="Путь к object_splits.parquet")
    parser.add_argument("-o", "--output", default="pairs_test",
                        help="Папка для результата (по умолчанию: pairs_test)")
    parser.add_argument("--text-only", action="store_true",
                        help="Генерировать только текст-текст пары")
    args = parser.parse_args()

    # --- Загрузка ---
    print("\n=== Загрузка таблиц ===")
    tables = load_tables(args)
    df_obj_params = tables["object_params"]
    df_object_splits = tables["object_splits"]

    print(f"\n  object_params:  {len(df_obj_params):>10,} строк")
    print(f"  object_splits:  {len(df_object_splits):>10,} строк")

    # --- Правила ---
    pair_rules = get_pair_rules(args.text_only)
    print(f"\n=== Правила генерации пар ({len(pair_rules)}) ===")
    for k1, k2, pt, both_text in pair_rules:
        tag = "TEXT×TEXT" if both_text else f"{k1}×{k2}"
        print(f"  • {k1:25} ↔ {k2:25} → {pt:25} [{tag}]")

    # --- Генерация ---
    print("\n=== Генерация позитивных пар ===")
    pos_df = generate_positive_pairs(df_obj_params, df_object_splits, pair_rules)
    print(f"  → {len(pos_df):,} позитивных пар")

    # --- Проверки ---
    print("\n=== Проверки целостности ===")
    checks = check_integrity(df_obj_params, df_object_splits, pos_df)
    all_ok = True
    for k, v in checks.items():
        if isinstance(v, list):
            print(f"  {k}: {v}")
        else:
            ok = "✅" if v == 0 else "❌"
            if v != 0:
                all_ok = False
            print(f"  {k}: {v} {ok}")

    if not all_ok:
        print("\n[!] Обнаружены проблемы целостности. См. отчёт.", file=sys.stderr)

    # --- Сохранение ---
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    print("\n=== Сохранение ===")
    target_cols = ["value_uid_1", "value_uid_2", "is_positive", "pair_type"]

    for split in ["train", "validation", "test"]:
        split_df = pos_df[pos_df["split"] == split][target_cols].copy()
        split_df = split_df.sample(frac=1, random_state=42).reset_index(drop=True)
        split_df["is_positive"] = split_df["is_positive"].astype("int8")

        path = output / f"{split}_pairs.parquet"
        split_df.to_parquet(path, index=False, engine="pyarrow")
        size_mb = path.stat().st_size / 1024 / 1024
        print(f"  → {path}  ({len(split_df):,} строк, {size_mb:.1f} MB)")

    # --- Отчёт ---
    report = build_report(pos_df, checks, args)
    report_path = output / "report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"\n  → {report_path}")

    print("\n🎉 Готово!")


if __name__ == "__main__":
    main()