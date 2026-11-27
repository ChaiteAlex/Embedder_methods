"""
Скрипт компиляции мультиязычного датасета пар.
Создает 4 типа пар из каждой исходной пары UID:
  - EN ↔ EN (монолингвальные английские)
  - RU ↔ RU (монолингвальные русские)
  - EN ↔ RU (кросс-лингвальные)
  - RU ↔ EN (кросс-лингвальные)
"""
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

# ============================================================
# КОНФИГУРАЦИЯ ТИПОВ ПАР
# ============================================================

# Какие комбинации языков создавать для каждой исходной пары
# Формат: (lang_1, lang_2, имя_типа)
LANG_PAIR_TYPES: List[Tuple[str, str, str]] = [
    ("text_en", "text_en", "en_en"),  # Английский ↔ Английский
    ("text_ru", "text_ru", "ru_ru"),  # Русский ↔ Русский
    ("text_en", "text_ru", "en_ru"),  # Английский ↔ Русский
    ("text_ru", "text_en", "ru_en"),  # Русский ↔ Английский
]


def build_uid_to_texts(df_texts: pd.DataFrame, df_params: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    """
    Строит словарь: uid -> {text_en: ..., text_ru: ..., fallback: ...}
    Для кодов (label, tactics) все языковые поля будут одинаковыми (сам код).
    """
    uid_map: Dict[str, Dict[str, str]] = {}

    # 1. Заполняем из texts.parquet (тут есть и text_en, и text_ru)
    for _, row in df_texts.iterrows():
        uid = row['value_uid']
        uid_map[uid] = {
            "text_en": str(row.get("text_en", "")).strip(),
            "text_ru": str(row.get("text_ru", "")).strip(),
            "correct": int(row.get("correct", 0)),
        }

    # 2. Дополняем из params.parquet (для кодов типа T1203, Execution и т.д.)
    for _, row in df_params.iterrows():
        uid = row['uid']
        if uid not in uid_map:
            value = str(row['value']).strip()
            # Для кодов используем одно и то же значение для обоих языков
            uid_map[uid] = {
                "text_en": value,
                "text_ru": value,
                "correct": 1,  # Коды считаем корректными по умолчанию
            }

    return uid_map


def expand_pairs_multilingual(
    df_pairs: pd.DataFrame,
    uid_map: Dict[str, Dict[str, str]],
    require_both_langs: bool = True,
) -> pd.DataFrame:
    """
    Разворачивает каждую пару UID в 4 языковые пары.
    
    Args:
        df_pairs: Исходный DataFrame с колонками value_uid_1, value_uid_2, is_positive, pair_type
        uid_map: Словарь uid -> {text_en, text_ru, correct}
        require_both_langs: Если True, пропускает пары, где хотя бы один текст пустой
    """
    expanded_rows = []
    skipped_stats = {
        "uid1_missing": 0,
        "uid2_missing": 0,
        "empty_text": 0,
        "low_quality": 0,
    }

    for _, row in df_pairs.iterrows():
        uid1 = row['value_uid_1']
        uid2 = row['value_uid_2']
        is_positive = row['is_positive']
        pair_type = row['pair_type']

        # Проверяем наличие UID в словаре
        if uid1 not in uid_map:
            skipped_stats["uid1_missing"] += 1
            continue
        if uid2 not in uid_map:
            skipped_stats["uid2_missing"] += 1
            continue

        texts1 = uid_map[uid1]
        texts2 = uid_map[uid2]

        # Создаем 4 типа пар
        for lang1, lang2, lang_pair_type in LANG_PAIR_TYPES:
            t1 = texts1[lang1]
            t2 = texts2[lang2]

            # Пропускаем пустые тексты
            if not t1 or not t2 or t1 == "nan" or t2 == "nan":
                skipped_stats["empty_text"] += 1
                continue

            # Опционально: пропускаем пары с некорректными переводами
            # (только для кросс-лингвальных пар, где качество критично)
            if require_both_langs and lang_pair_type in ("en_ru", "ru_en"):
                if texts1.get("correct", 0) == 0 or texts2.get("correct", 0) == 0:
                    skipped_stats["low_quality"] += 1
                    continue

            expanded_rows.append({
                "text_1": t1,
                "text_2": t2,
                "label": int(is_positive),
                "pair_type": pair_type,
                "lang_pair": lang_pair_type,
            })

    return pd.DataFrame(expanded_rows), skipped_stats


def main():
    parser = argparse.ArgumentParser(description="Компиляция мультиязычного датасета пар")
    parser.add_argument("--pairs-dir", required=True, help="Папка с train_pairs.parquet, test_pairs.parquet")
    parser.add_argument("--texts-path", required=True, help="Путь к texts.parquet")
    parser.add_argument("--params-path", required=True, help="Путь к params.parquet")
    parser.add_argument("--output-dir", required=True, help="Куда сохранить готовые датасеты")
    parser.add_argument("--mono-only", action="store_true", 
                        help="Создавать только монолингвальные пары (EN-EN, RU-RU)")
    parser.add_argument("--cross-only", action="store_true", 
                        help="Создавать только кросс-лингвальные пары (EN-RU, RU-EN)")
    parser.add_argument("--strict-quality", action="store_true",
                        help="Пропускать кросс-лингвальные пары с correct=0")
    args = parser.parse_args()

    pairs_dir = Path(args.pairs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Определяем, какие типы пар создавать
    if args.mono_only:
        lang_types = [t for t in LANG_PAIR_TYPES if t[0] == t[1]]
    elif args.cross_only:
        lang_types = [t for t in LANG_PAIR_TYPES if t[0] != t[1]]
    else:
        lang_types = LANG_PAIR_TYPES
    
    print(f"📋 Создаются типы пар: {[t[2] for t in lang_types]}")

    print("\n1. Загрузка справочников...")
    df_texts = pd.read_parquet(args.texts_path)
    df_params = pd.read_parquet(args.params_path)
    uid_map = build_uid_to_texts(df_texts, df_params)
    print(f"   Уникальных UID в словаре: {len(uid_map):,}")
    print(f"   Из них с обоими языками: {sum(1 for v in uid_map.values() if v['text_en'] and v['text_ru']):,}")

    splits = ["train", "validation", "test"]
    total_stats = {s: {"skipped": 0, "expanded": 0} for s in splits}

    print("\n2. Компиляция датасетов...")
    for split in splits:
        pairs_file = pairs_dir / f"{split}_pairs.parquet"
        if not pairs_file.exists():
            print(f"   [!] Пропуск: {pairs_file} не найден")
            continue

        print(f"\n   === Обработка {split} ===")
        df_pairs = pd.read_parquet(pairs_file)
        print(f"   Исходных пар: {len(df_pairs):,}")

        df_expanded, skipped = expand_pairs_multilingual(
            df_pairs, uid_map, require_both_langs=args.strict_quality
        )

        total_stats[split]["skipped"] = sum(skipped.values())
        total_stats[split]["expanded"] = len(df_expanded)

        print(f"   Пропущено: {sum(skipped.values()):,}")
        for k, v in skipped.items():
            if v > 0:
                print(f"      - {k}: {v:,}")

        # Статистика по типам языковых пар
        if len(df_expanded) > 0:
            print(f"   Распределение по типам пар:")
            for lang_pair, count in df_expanded['lang_pair'].value_counts().items():
                pct = count / len(df_expanded) * 100
                print(f"      {lang_pair:>6}: {count:>8,} ({pct:5.1f}%)")

            # Сохранение
            out_path = output_dir / f"{split}_dataset.parquet"
            df_expanded.to_parquet(out_path, index=False, engine="pyarrow")
            size_mb = out_path.stat().st_size / 1024 / 1024
            print(f"   ✅ Сохранено: {out_path} ({len(df_expanded):,} строк, {size_mb:.1f} MB)")

    # Итоговый отчет
    print("\n" + "=" * 60)
    print("ИТОГОВАЯ СТАТИСТИКА")
    print("=" * 60)
    print(f"{'Сплит':<12} {'Исходных':>12} {'Итоговых':>12} {'Пропущено':>12} {'Коэфф.':>8}")
    print("-" * 60)
    for split in splits:
        if total_stats[split]["expanded"] > 0 or total_stats[split]["skipped"] > 0:
            pairs_file = pairs_dir / f"{split}_pairs.parquet"
            if pairs_file.exists():
                orig_count = len(pd.read_parquet(pairs_file))
                expanded = total_stats[split]["expanded"]
                skipped = total_stats[split]["skipped"]
                ratio = expanded / orig_count if orig_count > 0 else 0
                print(f"{split:<12} {orig_count:>12,} {expanded:>12,} {skipped:>12,} {ratio:>7.1f}x")

    print("\n🎉 Компиляция мультиязычного датасета завершена!")
    print(f"   Файлы сохранены в: {output_dir}")
    print(f"\n💡 Для обучения используйте:")
    print(f"   python src/training/train_embedder.py \\")
    print(f"       --train {output_dir}/train_dataset.parquet \\")
    print(f"       --test {output_dir}/test_dataset.parquet \\")
    print(f"       --output-dir models/multilingual_embedder")


if __name__ == "__main__":
    main()