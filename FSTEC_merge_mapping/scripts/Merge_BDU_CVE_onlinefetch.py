"""
Скрипт объединения датасета CIRCL/Vulnerability-FSTEC (БДУ ФСТЭК)
с базой NVD (NIST) по прямым ссылкам на CVE.

Необходимые зависимости:

- datasets  - Загрузка датасета CIRCL/Vulnerability-FSTEC с Hugging Face
- pandas    - Работа с таблицами, explode, merge
- requests	- Запросы к API NVD
- tqdm	    - Прогресс-бар при скачивании NVD
- pyarrow   - Формат Parquet (зависимость datasets, ставится автоматически)

Автор: (ваш покорный слуга) - 1 промпт запрос Deepseek
Результат: merged_fstec_nvd.parquet + merged_fstec_nvd.csv
"""

import sys
import subprocess
import time
import os
from pathlib import Path

# =============================================================
# 1. ПРОВЕРКА И УСТАНОВКА ЗАВИСИМОСТЕЙ
# =============================================================

REQUIRED = {
    "datasets": "datasets",
    "pandas": "pandas",
    "requests": "requests",
    "tqdm": "tqdm",
}


def check_and_install():
    """Проверяет наличие библиотек, при отсутствии — предлагает установить."""
    missing = []
    for module, pip_name in REQUIRED.items():
        try:
            __import__(module)
            print(f"[OK]  {module} уже установлен")
        except ImportError:
            print(f"[!!]  {module} НЕ найден")
            missing.append(pip_name)

    if missing:
        print("\nОтсутствуют библиотеки:", ", ".join(missing))
        answer = input("Установить их сейчас через pip? [y/N]: ").strip().lower()
        if answer == "y":
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
            print("\nУстановка завершена. Перезапустите скрипт.")
        else:
            print("Без этих библиотек работа невозможна. Выход.")
        sys.exit(0)


check_and_install()

# Импортируем после проверки
import pandas as pd
import requests
from tqdm import tqdm
from datasets import load_dataset


# =============================================================
# 2. НАСТРОЙКИ
# =============================================================

NVD_API_KEY = os.getenv("NVD_API_KEY", "")  # можно задать переменной окружения
NVD_API_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Лимит запросов: с ключом — 50/30 сек, без ключа — 5/30 сек
REQUESTS_PER_PAGE = 2000  # максимум для NVD API v2
SLEEP_BETWEEN_CALLS = 6 if NVD_API_KEY else 7  # сек


# =============================================================
# 3. ЗАГРУЗКА ДАТАСЕТА FSTEC
# =============================================================

def load_fstec() -> pd.DataFrame:
    print("\n=== Шаг 1: Загрузка CIRCL/Vulnerability-FSTEC ===")
    ds = load_dataset("CIRCL/Vulnerability-FSTEC", split="train")
    df = ds.to_pandas()
    print(f"Загружено записей: {len(df)}")
    print(f"Колонки: {list(df.columns)}")
    return df


def explode_cve(df_fstec: pd.DataFrame) -> pd.DataFrame:
    """Разворачивает поле cve_id (строка с перечислением) в отдельные строки."""
    print("\n=== Шаг 2: Разворачивание поля cve_id ===")

    # Оставляем только непустые cve_id
    df = df_fstec[df_fstec["cve_id"].notna() & (df_fstec["cve_id"] != "")].copy()

    # Разделяем по запятой с пробелом
    df["cve_list"] = df["cve_id"].str.split(", ")
    df_exploded = df.explode("cve_list")

    # Оставляем только реальные CVE (убираем BID, RHSA, DSA и т.п.)
    df_exploded = df_exploded[
        df_exploded["cve_list"].str.match(r"^CVE-\d{4}-\d{4,}$", na=False)
    ].copy()

    df_exploded = df_exploded.rename(columns={"cve_list": "cve"})
    df_exploded = df_exploded.drop_duplicates(subset=["cve"])

    print(f"Уникальных CVE-идентификаторов: {len(df_exploded)}")
    return df_exploded


# =============================================================
# 4. ЗАГРУЗКА ДАННЫХ ИЗ NVD
# =============================================================

def fetch_nvd_by_cves(cve_list: list) -> pd.DataFrame:
    """
    Скачивает данные из NVD по списку CVE.
    Использует официальный API v2.0 с пагинацией.
    """
    print(f"\n=== Шаг 3: Загрузка NVD для {len(cve_list)} CVE ===")

    headers = {"apiKey": NVD_API_KEY} if NVD_API_KEY else {}
    records = []

    # NVD не поддерживает фильтр по списку CVE напрямую,
    # поэтому запрашиваем страницами и фильтруем по нужным ID.
    # Для ускорения можно фильтровать по дате, но здесь — общий подход.

    total = len(cve_list)
    cve_set = set(cve_list)

    start_index = 0
    pbar = tqdm(total=total, desc="NVD запросы")

    while True:
        params = {
            "resultsPerPage": REQUESTS_PER_PAGE,
            "startIndex": start_index,
        }
        try:
            r = requests.get(NVD_API_URL, headers=headers, params=params, timeout=60)
            r.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"\n[Ошибка] {e}. Пауза 30 сек и повтор...")
            time.sleep(30)
            continue

        data = r.json()
        vulnerabilities = data.get("vulnerabilities", [])
        if not vulnerabilities:
            break

        for item in vulnerabilities:
            cve_obj = item.get("cve", {})
            cve_id = cve_obj.get("id")
            if cve_id not in cve_set:
                continue

            # Извлекаем английское описание
            descriptions = cve_obj.get("descriptions", [])
            en_desc = next(
                (d["value"] for d in descriptions if d.get("lang") == "en"), ""
            )

            # Извлекаем метрики CVSS (берём любую доступную версию)
            metrics = cve_obj.get("metrics", {})
            cvss_score = None
            cvss_vector = None
            for key in ["cvssMetricV40", "cvssMetricV31", "cvssMetricV30", "cvssMetricV2"]:
                if key in metrics and metrics[key]:
                    m = metrics[key][0].get("cvssData", {})
                    cvss_score = m.get("baseScore")
                    cvss_vector = m.get("vectorString")
                    break

            # CPE (затронутое ПО)
            cpes = []
            for cfg in cve_obj.get("configurations", []):
                for node in cfg.get("nodes", []):
                    for cpe_match in node.get("cpeMatch", []):
                        if cpe_match.get("criteria"):
                            cpes.append(cpe_match["criteria"])

            records.append({
                "cve": cve_id,
                "nvd_description_en": en_desc,
                "nvd_cvss_score": cvss_score,
                "nvd_cvss_vector": cvss_vector,
                "nvd_cpes": cpes,
                "nvd_published": cve_obj.get("published"),
                "nvd_last_modified": cve_obj.get("lastModified"),
            })
            pbar.update(1)

        start_index += REQUESTS_PER_PAGE
        total_results = data.get("totalResults", 0)
        if start_index >= total_results:
            break

        time.sleep(SLEEP_BETWEEN_CALLS)

    pbar.close()
    df_nvd = pd.DataFrame(records)
    print(f"Скачано записей из NVD: {len(df_nvd)}")
    return df_nvd


# =============================================================
# 5. ОБЪЕДИНЕНИЕ
# =============================================================

def merge_datasets(df_fstec: pd.DataFrame, df_nvd: pd.DataFrame) -> pd.DataFrame:
    print("\n=== Шаг 4: JOIN по полю CVE ===")
    merged = pd.merge(df_fstec, df_nvd, on="cve", how="inner")
    print(f"Записей после JOIN: {len(merged)}")
    return merged


# =============================================================
# 6. СОХРАНЕНИЕ
# =============================================================

def save_results(df: pd.DataFrame):
    print("\n=== Шаг 5: Сохранение ===")
    parquet_path = OUTPUT_DIR / "merged_fstec_nvd.parquet"
    csv_path = OUTPUT_DIR / "merged_fstec_nvd.csv"

    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"[OK] Parquet: {parquet_path}")
    print(f"[OK] CSV:     {csv_path}")
    print(f"Итоговый размер: {len(df)} строк, {len(df.columns)} колонок")


# =============================================================
# 7. MAIN
# =============================================================

def main():
    df_fstec_raw = load_fstec()
    df_fstec = explode_cve(df_fstec_raw)

    cve_list = df_fstec["cve"].tolist()
    df_nvd = fetch_nvd_by_cves(cve_list)

    if df_nvd.empty:
        print("\n[!] NVD не вернул данных. Проверьте интернет или API-ключ.")
        return

    merged = merge_datasets(df_fstec, df_nvd)
    save_results(merged)


if __name__ == "__main__":
    main()