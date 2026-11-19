"""
Сборка датасета пар для обучения эмбеддеров из MITRE-таблиц.

Примеры запуска:
    # Базовый запуск (таблицы в папке mitre_tables/, результат туда же)
    python build_pairs.py -i mitre_tables/ -o pairs_out/

    # Явно указать пути к parquet-файлам
    python build_pairs.py \
        --params mitre_tables/params.parquet \
        --object-params mitre_tables/object_params.parquet \
        --object-splits mitre_tables/object_splits.parquet \
        --texts mitre_tables/texts.parquet \
        -o pairs_out/

    # Только текст-текст пары (без текст-код)
    python build_pairs.py -i mitre_tables/ -o pairs_out/ --text-only

    # Задать соотношение негативов к позитивам (по умолчанию 1:1)
    python build_pairs.py -i mitre_tables/ -o pairs_out/ --neg-ratio 2.0

    # Фиксированный random seed для воспроизводимости
    python build_pairs.py -i mitre_tables/ -o pairs_out/ --seed 42
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any

import pandas as pd
import numpy as np


# ============================================================
# КОНСТАНТЫ
# ============================================================

# Ключи, которые являются ТЕКСТАМИ (есть в texts.parquet)
TEXT_KEYS = {"event_text", "technique_description", "technique_name"}

# Ключи, которые являются КОДАМИ/МЕТКАМИ (нет в texts.parquet)
CODE_KEYS = {"label", "parent_label", "tactics"}

# Правила генерации пар: (key_1, key_2, pair_type, both_text)
# both_text=True → пара идёт и в позитивы, и в негативы (для эмбеддеров)
# both_text=False → пара идёт только в позитивы (для мульти-модальности)
PAIR_RULES: List[Tuple[str, str, str, bool]] = [
    # Текст ↔ Текст (основные пары для контрастивного обучения)
    ("event_text", "technique_description", "eve_to_tec_desc", True),
    ("event_text", "technique_name", "eve_to_tec_name", True),
    ("technique_name", "technique_description", "tec_name_to_desc", True),

    # Текст ↔ Код (опционально, для связывания эмбеддера с метками)
    ("event_text", "label", "eve_to_label", False),
    ("event_text", "tactics", "eve_to_tactics", False),
    ("technique_description", "label", "tec_desc_to_label", False),
    ("technique_description", "tactics", "tec_desc_to_tactics", False),
    ("technique_name", "label", "tec_name_to_label", False),
    ("technique_name", "tactics", "tec_name_to_tactics", False),

    # Код ↔ Код (структурные связи, не для эмбеддеров)
    ("label", "parent_label", "label_to_parent", False),
    ("label", "tactics", "label_to_tactics", False),
]


# ============================================================
# ВВОД
# ============================================================

def load_tables(args) -> Dict[str, pd.DataFrame]:
    """Загружает parquet-таблицы по явным путям или из папки."""
    if args.inputs:
        base = Path(args.inputs)
        if not base.is_dir():
            print(f"[!] Не папка: {base}", file=sys.stderr)
            sys.exit(1)
        paths = {
            "params": base / "params.parquet",
            "object_params": base / "object_params.parquet",
            "object_splits": base / "object_splits.parquet",
            "texts": base / "texts.parquet",
        }
    else:
        paths = {
            "params": Path(args.params),
            "object_params": Path(args.object_params),
            "object_splits": Path(args.object_splits),
            "texts": Path(args.texts),
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
    """Фильтрует правила в зависимости от флага text_only."""
    if text_only:
        return [r for r in PAIR_RULES if r[3]]
    return PAIR_RULES


def generate_positive_pairs(
    df_obj_params: pd.DataFrame,
    df_object_splits: pd.DataFrame,
    pair_rules: List[Tuple[str, str, str, bool]],
) -> pd.DataFrame:
    """
    Генерирует позитивные пары: все комбинации параметров внутри одного объекта.
    """
    # Присоединяем split к object_params
    df = df_obj_params.merge(
        df_object_splits[["object_uid", "split"]],
        on="object_uid",
        how="inner",
    )

    all_rows: List[Dict[str, Any]] = []

    # Группируем по (object_uid, split) — это быстрее, чем apply
    grouped = df.groupby(["object_uid", "split"])

    for (obj_uid, split), group in grouped:
        # Словарь: key → список uid
        key_to_uids = group.groupby("key")["uid"].apply(list).to_dict()

        for k1, k2, pair_type, _ in pair_rules:
            list1 = key_to_uids.get(k1, [])
            list2 = key_to_uids.get(k2, [])
            if not list1 or not list2:
                continue

            # Декартово произведение
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


def generate_negative_pairs(
    pos_df: pd.DataFrame,
    neg_ratio: float,
    seed: int,
) -> pd.DataFrame:
    """
    Генерирует негативные пары путём циклического сдвига внутри (split, pair_type).
    Негативы берутся ТОЛЬКО из текст-текст пар.
    """
    rng = np.random.default_rng(seed)

    # Берём только текст-текст пары для негативов
    text_pair_types = {r[2] for r in PAIR_RULES if r[3]}
    source = pos_df[pos_df["pair_type"].isin(text_pair_types)].copy()

    if source.empty:
        print("[!] Нет текст-текст пар для генерации негативов.")
        return pd.DataFrame(columns=["value_uid_1", "value_uid_2", "pair_type", "is_positive", "split"])

    # Перемешиваем внутри каждой группы (split, pair_type) для разнообразия
    source = source.groupby(["split", "pair_type"], group_keys=False).apply(
        lambda g: g.sample(frac=1, random_state=rng.integers(0, 2**31))
    )

    neg_parts = []
    for _, group in source.groupby(["split", "pair_type"]):
        n_target = int(len(group) * neg_ratio)
        if n_target == 0:
            continue

        # Циклический сдвиг: берём uid2 из "соседних" объектов
        neg = group[["value_uid_1", "value_uid_2", "pair_type", "split"]].copy()

        # Повторяем сдвиг, пока не наберём нужное количество
        while len(neg) < n_target:
            shifted = neg.copy()
            shifted["value_uid_2"] = shifted["value_uid_2"].shift(-1)
            # Заполняем последний элемент первым (циклически)
            shifted.iloc[-1, shifted.columns.get_loc("value_uid_2")] = neg.iloc[0]["value_uid_2"]
            neg = pd.concat([neg, shifted], ignore_index=True)

        neg = neg.head(n_target)

        # Исключаем случайные совпадения (uid1 == uid2)
        neg = neg[neg["value_uid_1"] != neg["value_uid_2"]]
        neg["is_positive"] = 0
        neg_parts.append(neg)

    if not neg_parts:
        return pd.DataFrame(columns=["value_uid_1", "value_uid_2", "pair_type", "is_positive", "split"])

    return pd.concat(neg_parts, ignore_index=True)


# ============================================================
# ПРОВЕРКИ ЦЕЛОСТНОСТИ
# ============================================================

def check_integrity(
    df_params: pd.DataFrame,
    df_obj_params: pd.DataFrame,
    df_object_splits: pd.DataFrame,
    df_texts: pd.DataFrame,
    pos_df: pd.DataFrame,
    neg_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Проверяет целостность данных после генерации пар."""
    checks: Dict[str, Any] = {}

    # UID из пар должны существовать в params
    pair_uids = set(pos_df["value_uid_1"].unique()) | set(pos_df["value_uid_2"].unique())
    pair_uids |= set(neg_df["value_uid_1"].unique()) | set(neg_df["value_uid_2"].unique())
    params_uids = set(df_params["uid"].unique())
    checks["pair_uids_missing_in_params"] = len(pair_uids - params_uids)

    # Негативы не должны содержать позитивные пары
    pos_set = set(zip(pos_df["value_uid_1"], pos_df["value_uid_2"], pos_df["pair_type"]))
    neg_set = set(zip(neg_df["value_uid_1"], neg_df["value_uid_2"], neg_df["pair_type"]))
    checks["neg_overlaps_with_pos"] = len(pos_set & neg_set)

    # В негативах uid1 != uid2
    checks["neg_self_pairs"] = int((neg_df["value_uid_1"] == neg_df["value_uid_2"]).sum())

    # Утечка сплитов: негативы из train не должны содержать uid из test/val
    for split in ["train", "validation", "test"]:
        train_uids = set(df_obj_params[
            df_obj_params["object_uid"].isin(
                df_object_splits[df_object_splits["split"] == split]["object_uid"]
            )
        ]["uid"].unique())

        neg_in_split = neg_df[neg_df["split"] == split]
        neg_uids = set(neg_in_split["value_uid_1"].unique()) | set(neg_in_split["value_uid_2"].unique())
        leaks = neg_uids - train_uids
        checks[f"neg_{split}_leaks_from_other_splits"] = len(leaks)

    # Дубликаты в парах
    checks["pos_duplicates"] = int(pos_df.duplicated(subset=["value_uid_1", "value_uid_2", "pair_type"]).sum())
    checks["neg_duplicates"] = int(neg_df.duplicated(subset=["value_uid_1", "value_uid_2", "pair_type"]).sum())

    # Сплиты в парах
    checks["splits_in_pairs"] = sorted(pos_df["split"].unique().tolist())

    return checks


# ============================================================
# ОТЧЁТ
# ============================================================

def build_report(
    df_params: pd.DataFrame,
    df_object_splits: pd.DataFrame,
    df_texts: pd.DataFrame,
    pos_df: pd.DataFrame,
    neg_df: pd.DataFrame,
    checks: Dict[str, Any],
    args,
) -> str:
    """Строит markdown-отчёт по сборке датасета."""
    lines = ["# Отчёт по сборке датасета пар\n"]

    # Параметры запуска
    lines.append("## Параметры запуска\n")
    lines.append(f"- **text_only**: `{args.text_only}`")
    lines.append(f"- **neg_ratio**: `{args.neg_ratio}`")
    lines.append(f"- **seed**: `{args.seed}`")
    lines.append("")

    # Сводка по парам
    lines.append("## Сводка по парам\n")
    lines.append("| Сплит | Позитивных | Негативных | Всего | Поз/Нег |")
    lines.append("|-|-:|-:|-:|-:|")
    for split in ["train", "validation", "test"]:
        pos_n = len(pos_df[pos_df["split"] == split])
        neg_n = len(neg_df[neg_df["split"] == split])
        total = pos_n + neg_n
        ratio = f"{pos_n / neg_n:.2f}" if neg_n > 0 else "∞"
        lines.append(f"| `{split}` | {pos_n:,} | {neg_n:,} | {total:,} | {ratio} |")
    lines.append("")

    # Распределение по pair_type
    lines.append("## Распределение по `pair_type`\n")
    lines.append("| pair_type | Позитивных | Негативных | Всего |")
    lines.append("|-|-:|-:|-:|")
    all_pairs = pd.concat([pos_df, neg_df], ignore_index=True)
    for pt, group in all_pairs.groupby("pair_type"):
        pos_n = (group["is_positive"] == 1).sum()
        neg_n = (group["is_positive"] == 0).sum()
        lines.append(f"| `{pt}` | {pos_n:,} | {neg_n:,} | {len(group):,} |")
    lines.append("")

    # Статистика по текстам (для контекста)
    lines.append("## Статистика по `texts` (для кросс-валидации)\n")
    lines.append("| source_key | Всего | correct=1 | correct=0 |")
    lines.append("|-|-:|-:|-:|")
    for sk, group in df_texts.groupby("source_key"):
        c1 = (group["correct"] == 1).sum()
        c0 = (group["correct"] == 0).sum()
        lines.append(f"| `{sk}` | {len(group):,} | {c1:,} | {c0:,} |")
    lines.append("")

    # Проверки целостности
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
    lines.append("### Позитивные\n")
    lines.append("```")
    lines.append(pos_df.head(5).to_string(index=False))
    lines.append("```\n")
    lines.append("### Негативные\n")
    lines.append("```")
    lines.append(neg_df.head(5).to_string(index=False))
    lines.append("```\n")

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Сборка датасета пар для обучения эмбеддеров",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-i", "--inputs", help="Папка с parquet-таблицами")
    group.add_argument("--params", help="Путь к params.parquet")

    parser.add_argument("--object-params", help="Путь к object_params.parquet")
    parser.add_argument("--object-splits", help="Путь к object_splits.parquet")
    parser.add_argument("--texts", help="Путь к texts.parquet")
    parser.add_argument("-o", "--output", default="pairs_out",
                        help="Папка для результата (по умолчанию: pairs_out)")
    parser.add_argument("--text-only", action="store_true",
                        help="Генерировать только текст-текст пары")
    parser.add_argument("--neg-ratio", type=float, default=1.0,
                        help="Соотношение негативов к позитивам (по умолчанию: 1.0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (по умолчанию: 42)")
    args = parser.parse_args()

    # --- Загрузка ---
    print("\n=== Загрузка таблиц ===")
    tables = load_tables(args)
    df_params = tables["params"]
    df_obj_params = tables["object_params"]
    df_object_splits = tables["object_splits"]
    df_texts = tables["texts"]

    print(f"\n  params:         {len(df_params):>10,} строк")
    print(f"  object_params:  {len(df_obj_params):>10,} строк")
    print(f"  object_splits:  {len(df_object_splits):>10,} строк")
    print(f"  texts:          {len(df_texts):>10,} строк")

    # --- Правила ---
    pair_rules = get_pair_rules(args.text_only)
    print(f"\n=== Правила генерации пар ({len(pair_rules)}) ===")
    for k1, k2, pt, both_text in pair_rules:
        tag = "TEXT×TEXT" if both_text else f"{k1}×{k2}"
        print(f"  • {k1:25} ↔ {k2:25} → {pt:25} [{tag}]")

    # --- Позитивные пары ---
    print("\n=== Генерация позитивных пар ===")
    pos_df = generate_positive_pairs(df_obj_params, df_object_splits, pair_rules)
    print(f"  → {len(pos_df):,} позитивных пар")

    # --- Негативные пары ---
    print(f"\n=== Генерация негативных пар (ratio={args.neg_ratio}) ===")
    neg_df = generate_negative_pairs(pos_df, args.neg_ratio, args.seed)
    print(f"  → {len(neg_df):,} негативных пар")

    # --- Проверки ---
    print("\n=== Проверки целостности ===")
    checks = check_integrity(df_params, df_obj_params, df_object_splits, df_texts, pos_df, neg_df)
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
        split_pos = pos_df[pos_df["split"] == split][target_cols].copy()
        split_neg = neg_df[neg_df["split"] == split][target_cols].copy()
        split_all = pd.concat([split_pos, split_neg], ignore_index=True)

        # Перемешиваем для равномерного распределения в батчах
        split_all = split_all.sample(frac=1, random_state=args.seed).reset_index(drop=True)

        # Приводим типы для экономии места
        split_all["is_positive"] = split_all["is_positive"].astype(np.int8)

        path = output / f"{split}_pairs.parquet"
        split_all.to_parquet(path, index=False, engine="pyarrow")
        size_mb = path.stat().st_size / 1024 / 1024
        print(f"  → {path}  ({len(split_all):,} строк, {size_mb:.1f} MB)")

    # --- Отчёт ---
    report = build_report(df_params, df_object_splits, df_texts, pos_df, neg_df, checks, args)
    report_path = output / "report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"\n  → {report_path}")

    print("\n🎉 Готово!")


if __name__ == "__main__":
    main()