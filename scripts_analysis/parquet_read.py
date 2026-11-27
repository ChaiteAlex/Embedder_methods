"""
Удобная читалка Parquet-файлов с графом BDU ↔ CVE + unmatched.
"""

import sys
import subprocess
import argparse
from pathlib import Path
from collections import Counter

# =============================================================
# ПРОВЕРКА ЗАВИСИМОСТЕЙ
# =============================================================

REQUIRED = {"pandas": "pandas", "pyarrow": "pyarrow"}


def check_and_install():
    missing = [p for m, p in REQUIRED.items() if not _has_module(m)]
    if missing:
        print("Отсутствуют библиотеки:", ", ".join(missing))
        if input("Установить? [y/N]: ").strip().lower() == "y":
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
        else:
            sys.exit(1)


def _has_module(name):
    try:
        __import__(name)
        return True
    except ImportError:
        return False


check_and_install()

import pandas as pd
pd.set_option("display.max_colwidth", 120)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)


# =============================================================
# НАСТРОЙКИ
# =============================================================

PARQUET_DIR = Path("output")
BDU_FILE = PARQUET_DIR / "bdu_to_cve.parquet"
CVE_FILE = PARQUET_DIR / "cve_to_bdu.parquet"
UNMATCHED_FILE = PARQUET_DIR / "unmatched.parquet"

SEPARATOR = "=" * 80


# =============================================================
# ЗАГРУЗКА
# =============================================================

def load_data():
    missing = [f for f in [BDU_FILE, CVE_FILE] if not f.exists()]
    if missing:
        print(f"[!] Не найдены файлы: {[str(f) for f in missing]}")
        sys.exit(1)

    df_bdu = pd.read_parquet(BDU_FILE)
    df_cve = pd.read_parquet(CVE_FILE)

    df_unm = None
    if UNMATCHED_FILE.exists():
        df_unm = pd.read_parquet(UNMATCHED_FILE)

    return df_bdu, df_cve, df_unm


# =============================================================
# ОБЗОР
# =============================================================

def show_overview(df_bdu, df_cve, df_unm=None):
    print(SEPARATOR)
    print("ОБЗОР PARQUET-ФАЙЛОВ")
    print(SEPARATOR)

    print(f"\n📁 {BDU_FILE}")
    print(f"   Строк:   {len(df_bdu)}")
    print(f"   Поля:    {list(df_bdu.columns)}")

    print(f"\n📁 {CVE_FILE}")
    print(f"   Строк:   {len(df_cve)}")
    print(f"   Поля:    {list(df_cve.columns)}")

    if df_unm is not None:
        print(f"\n📁 {UNMATCHED_FILE}")
        print(f"   Строк:   {len(df_unm)}")
        print(f"   Поля:    {list(df_unm.columns)}")
        print(f"\n   Разбивка по типу:")
        for t, count in df_unm["type"].value_counts().items():
            print(f"     {t:25s} {count}")

    print(f"\n--- Первые 3 записи BDU ---")
    print(df_bdu[["bdu_id", "title", "cve_list", "nvd_available"]].head(3).to_string(index=False))

    print(f"\n--- Первые 3 записи CVE ---")
    print(df_cve[["cve_id", "bdu_list", "nvd_cvss_score"]].head(3).to_string(index=False))


# =============================================================
# STATS
# =============================================================

def show_stats(df_bdu, df_cve, df_unm=None):
    print(SEPARATOR)
    print("СТАТИСТИКА")
    print(SEPARATOR)

    total_bdu = len(df_bdu)
    bdu_with_cve = df_bdu["cve_list"].apply(lambda x: len(x) > 0).sum()
    bdu_nvd_avail = df_bdu["nvd_available"].sum()

    print(f"\n📊 BDU:")
    print(f"   Всего:                 {total_bdu}")
    print(f"   С хотя бы одной CVE:   {bdu_with_cve}")
    print(f"   Без CVE (RU-only):     {total_bdu - bdu_with_cve}")
    print(f"   Есть данные в NVD:     {bdu_nvd_avail}")

    cve_counts = df_bdu["cve_list"].apply(len)
    print(f"\n   Распределение по числу CVE:")
    print(f"     1 CVE:  {(cve_counts == 1).sum()}")
    print(f"     2-5:    {((cve_counts >= 2) & (cve_counts <= 5)).sum()}")
    print(f"     6-10:   {((cve_counts >= 6) & (cve_counts <= 10)).sum()}")
    print(f"     11+:    {(cve_counts > 10).sum()}")

    total_cve = len(df_cve)
    cve_with_bdu = df_cve["bdu_list"].apply(lambda x: len(x) > 0).sum()
    cve_with_nvd = df_cve["nvd_description_en"].apply(lambda x: bool(str(x).strip())).sum()

    print(f"\n📊 CVE:")
    print(f"   Всего:                 {total_cve}")
    print(f"   Со ссылкой на BDU:     {cve_with_bdu}")
    print(f"   С описанием из NVD:    {cve_with_nvd}")

    cvss = df_cve["nvd_cvss_score"].dropna()
    if len(cvss) > 0:
        print(f"\n   CVSS (из NVD):")
        print(f"     Заполнено:        {len(cvss)}")
        print(f"     Мин / Макс:       {cvss.min():.1f} / {cvss.max():.1f}")
        print(f"     Среднее:          {cvss.mean():.2f}")
        print(f"     Критичных (9.0+): {(cvss >= 9.0).sum()}")
        print(f"     Высоких (7.0-8.9): {((cvss >= 7.0) & (cvss < 9.0)).sum()}")

    # Unmatched
    if df_unm is not None:
        print(f"\n📊 UNMATCHED:")
        for t, count in df_unm["type"].value_counts().items():
            print(f"   {t:25s} {count}")
        print(f"   Всего:                    {len(df_unm)}")

    # Топ-10 вендоров
    print(f"\n📊 Топ-10 вендоров (по числу CPE):")
    all_cpes = []
    for cpes in df_cve["nvd_cpes"].dropna():
        if isinstance(cpes, (list, tuple)):
            all_cpes.extend(cpes)

    if all_cpes:
        vendors = []
        for cpe in all_cpes:
            parts = cpe.split(":")
            if len(parts) > 3:
                vendors.append(parts[3])
        for vendor, count in Counter(vendors).most_common(10):
            print(f"     {vendor:30s} {count}")


# =============================================================
# UNMATCHED — ПРОСМОТР
# =============================================================

def show_unmatched(df_unm: pd.DataFrame, limit: int = 30):
    """
    Показывает unmatched-записи, сгруппированные по типу.
    Структура: колонки 'type' и 'id'.
    """
    print(SEPARATOR)
    print("UNMATCHED ЗАПИСИ")
    print(SEPARATOR)

    if df_unm is None:
        print("[!] Файл unmatched.parquet не загружен")
        return

    for t in df_unm["type"].unique():
        subset = df_unm[df_unm["type"] == t]
        print(f"\n📊 {t}  —  всего {len(subset)}")

        # Группируем по году для наглядности
        def get_year(s):
            # BDU:2024-04519 → 2024 ; CVE-2024-3584 → 2024
            try:
                return str(s).split(":")[1].split("-")[0]
            except Exception:
                return "?"

        years = subset["id"].apply(get_year)
        year_counts = Counter(years).most_common()
        print(f"   По годам:")
        for year, count in year_counts[:15]:
            print(f"     {year}: {count}")

        print(f"\n   Первые {min(limit, len(subset))} записей:")
        for i, row in subset.head(limit).iterrows():
            print(f"     {row['id']}")


def find_unmatched_id(df_unm: pd.DataFrame, target_id: str):
    """Проверяет, есть ли конкретный ID в unmatched."""
    target_id = target_id.strip()
    row = df_unm[df_unm["id"] == target_id]

    if row.empty:
        print(f"[i] {target_id} НЕ в unmatched (значит, у него есть связь)")
        return

    print(f"[!] {target_id} в unmatched, тип: {row.iloc[0]['type']}")


# =============================================================
# ПОИСК (BDU / CVE / текст / вендор)
# =============================================================

def find_by_bdu(df_bdu, df_cve, df_unm, bdu_id: str):
    bdu_id = bdu_id.strip()
    row = df_bdu[df_bdu["bdu_id"] == bdu_id]

    if row.empty:
        print(f"[!] BDU {bdu_id} не найден")
        if df_unm is not None:
            find_unmatched_id(df_unm, bdu_id)
        return

    row = row.iloc[0]
    print(SEPARATOR)
    print(f"BDU: {bdu_id}")
    print(SEPARATOR)
    print(f"\nTitle:       {row['title']}")
    print(f"\nDescription: {row['description']}")
    print(f"\nCVSS v2.0:   {row['cvss_v2_0_fstec']}")
    print(f"NVD доступно: {row['nvd_available']}")
    print(f"CVE:         {row['cve_list']}")

    if row["cve_list"]:
        print(f"\n--- Связанные CVE из NVD ---")
        related = df_cve[df_cve["cve_id"].isin(row["cve_list"])]
        for _, cve_row in related.iterrows():
            print(f"\n  🔹 {cve_row['cve_id']}")
            print(f"     CVSS: {cve_row['nvd_cvss_score']}  ({cve_row['nvd_cvss_vector']})")
            desc = cve_row["nvd_description_en"]
            print(f"     EN:   {desc[:200]}{'...' if len(desc) > 200 else ''}")


def find_by_cve(df_bdu, df_cve, df_unm, cve_id: str):
    cve_id = cve_id.strip()
    row = df_cve[df_cve["cve_id"] == cve_id]

    if row.empty:
        print(f"[!] CVE {cve_id} не найден")
        if df_unm is not None:
            find_unmatched_id(df_unm, cve_id)
        return

    row = row.iloc[0]
    print(SEPARATOR)
    print(f"CVE: {cve_id}")
    print(SEPARATOR)
    print(f"\nCVSS:         {row['nvd_cvss_score']}")
    print(f"Вектор:       {row['nvd_cvss_vector']}")
    print(f"Опубликовано: {row['nvd_published']}")

    print(f"\nОписание EN:\n{row['nvd_description_en']}")

    if row["nvd_cpes"]:
        print(f"\nCPE ({len(row['nvd_cpes'])} шт.):")
        for cpe in row["nvd_cpes"][:10]:
            print(f"  {cpe}")
        if len(row["nvd_cpes"]) > 10:
            print(f"  ... и ещё {len(row['nvd_cpes']) - 10}")

    if row["bdu_list"]:
        print(f"\n--- Связанные BDU ---")
        related = df_bdu[df_bdu["bdu_id"].isin(row["bdu_list"])]
        for _, bdu_row in related.iterrows():
            print(f"\n  🔹 {bdu_row['bdu_id']}")
            print(f"     {bdu_row['title']}")


def search_text(df_bdu, df_cve, query: str, limit: int = 10):
    q = query.lower()
    print(SEPARATOR)
    print(f"Поиск: '{query}'")
    print(SEPARATOR)

    mask_bdu = (
        df_bdu["title"].str.lower().str.contains(q, na=False) |
        df_bdu["description"].str.lower().str.contains(q, na=False)
    )
    if mask_bdu.any():
        print(f"\n--- В BDU ({mask_bdu.sum()} найдено) ---")
        for _, row in df_bdu[mask_bdu].head(limit).iterrows():
            print(f"\n  🔹 {row['bdu_id']}")
            print(f"     {row['title'][:150]}")

    mask_cve = df_cve["nvd_description_en"].str.lower().str.contains(q, na=False)
    if mask_cve.any():
        print(f"\n--- В CVE ({mask_cve.sum()} найдено) ---")
        for _, row in df_cve[mask_cve].head(limit).iterrows():
            print(f"\n  🔹 {row['cve_id']}  CVSS: {row['nvd_cvss_score']}")
            print(f"     {row['nvd_description_en'][:150]}")


def filter_by_vendor(df_cve, vendor: str, limit: int = 15):
    v = vendor.lower()
    print(SEPARATOR)
    print(f"Фильтр по вендору: '{vendor}'")
    print(SEPARATOR)

    def has_vendor(cpes):
        if not isinstance(cpes, (list, tuple)):
            return False
        return any(
            len(cpe.split(":")) > 3 and cpe.split(":")[3].lower() == v
            for cpe in cpes
        )

    mask = df_cve["nvd_cpes"].apply(has_vendor)
    found = df_cve[mask]
    if found.empty:
        print(f"[!] Ничего не найдено для '{vendor}'")
        return

    print(f"\nНайдено: {mask.sum()} (показано {min(limit, len(found))})")
    for _, row in found.head(limit).iterrows():
        print(f"\n  🔹 {row['cve_id']}  CVSS: {row['nvd_cvss_score']}")
        print(f"     {row['nvd_description_en'][:150]}")
        print(f"     BDU: {row['bdu_list']}")


# =============================================================
# MAIN
# =============================================================

def main():
    parser = argparse.ArgumentParser(description="Читалка Parquet BDU ↔ CVE")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--bdu", type=str)
    parser.add_argument("--cve", type=str)
    parser.add_argument("--search", type=str)
    parser.add_argument("--vendor", type=str)
    parser.add_argument("--unmatched", action="store_true",
                        help="Только unmatched-записи")
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()

    df_bdu, df_cve, df_unm = load_data()

    if args.unmatched:
        show_unmatched(df_unm, limit=args.limit)
    elif args.stats:
        show_stats(df_bdu, df_cve, df_unm)
    elif args.bdu:
        find_by_bdu(df_bdu, df_cve, df_unm, args.bdu)
    elif args.cve:
        find_by_cve(df_bdu, df_cve, df_unm, args.cve)
    elif args.search:
        search_text(df_bdu, df_cve, args.search, args.limit)
    elif args.vendor:
        filter_by_vendor(df_cve, args.vendor, args.limit)
    else:
        show_overview(df_bdu, df_cve, df_unm)
        print("\n" + SEPARATOR)
        print("Подсказки:")
        print("  python read_parquet.py --stats")
        print("  python read_parquet.py --unmatched")
        print("  python read_parquet.py --bdu BDU:2024-04519")
        print("  python read_parquet.py --cve CVE-2024-3584")
        print("  python read_parquet.py --search 'Microsoft Office'")
        print("  python read_parquet.py --vendor microsoft")
        print(SEPARATOR)


if __name__ == "__main__":
    main()