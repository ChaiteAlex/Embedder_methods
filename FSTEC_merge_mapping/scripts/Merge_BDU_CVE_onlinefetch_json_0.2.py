"""
Построение JSON + Parquet графа связей BDU ↔ CVE.
Источники: CIRCL/Vulnerability-FSTEC + NVD (локальный дамп ИЛИ API).
"""

import sys
import subprocess
import json
import gzip
import lzma
import re
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# =============================================================
# 1. ПРОВЕРКА ЗАВИСИМОСТЕЙ
# =============================================================

REQUIRED = {
    "datasets": "datasets",
    "pandas": "pandas",
    "tqdm": "tqdm",
    "requests": "requests",
    "pyarrow": "pyarrow",
}


def check_and_install():
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

import pandas as pd
import requests
from tqdm import tqdm
from datasets import load_dataset


# =============================================================
# 2. НАСТРОЙКИ
# =============================================================

# API NVD
NVD_API_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
NVD_API_KEY = "96FFB8DD-90AF-48D8-8900-54AF8F58B20C"  # ← вставьте ключ, если есть (иначе оставьте пустым)
NVD_DUMP_DIR = Path("./nvd_dumps")
USE_API_IF_NO_DUMP = True  # если папки с дампом нет — скачать через API

# Выходные файлы
OUTPUT_JSON = Path("bdu_cve_mapping.json")
OUTPUT_PARQUET_DIR = Path("./parquet_output")

CVE_PATTERN = re.compile(r"^CVE-\d{4}-\d{4,}$")


# =============================================================
# 3. ЗАГРУЗКА FSTEC
# =============================================================

def load_fstec() -> pd.DataFrame:
    print("\n=== Шаг 1: Загрузка CIRCL/Vulnerability-FSTEC ===")
    ds = load_dataset("CIRCL/Vulnerability-FSTEC", split="train", token="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")
    df = ds.to_pandas()
    print(f"Загружено записей: {len(df)}")
    return df


# =============================================================
# 4. СКАЧИВАНИЕ ЧЕРЕЗ API NVD
# =============================================================

def fetch_nvd_via_api() -> dict:
    """
    Скачивает все CVE через официальный API NVD v2.0.
    Возвращает словарь {cve_id: {...}}.
    """
    print("\n=== Шаг 2a: Скачивание NVD через API ===")

    headers = {"apiKey": NVD_API_KEY} if NVD_API_KEY else {}
    results_per_page = 2000  # максимум для NVD API v2
    # Задержка между запросами: с ключом 0.6 сек, без — 6 сек [citation:6]
    sleep_between = 0.6 if NVD_API_KEY else 6.0

    nvd_index = {}
    start_index = 0
    total_results = None

    pbar = None
    while True:
        params = {
            "resultsPerPage": results_per_page,
            "startIndex": start_index,
        }

        try:
            r = requests.get(NVD_API_URL, headers=headers, params=params, timeout=120)
            r.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"\n[Ошибка] {e}. Пауза 30 сек и повтор...")
            time.sleep(30)
            continue

        data = r.json()
        vulnerabilities = data.get("vulnerabilities", [])

        # Инициализируем прогресс-бар после первого ответа
        if total_results is None:
            total_results = data.get("totalResults", 0)
            print(f"Всего CVE в NVD: {total_results}")
            pbar = tqdm(total=total_results, desc="Скачивание NVD")

        if not vulnerabilities:
            break

        for item in vulnerabilities:
            parsed = parse_nvd_item(item)
            if parsed:
                nvd_index[parsed["cve"]] = parsed
                if pbar:
                    pbar.update(1)

        start_index += results_per_page
        if start_index >= total_results:
            break

        time.sleep(sleep_between)

    if pbar:
        pbar.close()

    print(f"Скачано записей из NVD: {len(nvd_index)}")
    return nvd_index


# =============================================================
# 5. ЧТЕНИЕ ЛОКАЛЬНОГО ДАМПА NVD
# =============================================================

def read_json_auto(path: Path) -> dict:
    name = path.name.lower()
    if name.endswith(".xz"):
        with lzma.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    elif name.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_nvd_item(item: dict) -> dict | None:
    """Извлекает нужные поля из одной записи NVD (формат API v2.0)."""
    cve_obj = item.get("cve", {})
    cve_id = cve_obj.get("id")
    if not cve_id:
        return None

    descriptions = cve_obj.get("descriptions", [])
    en_desc = next((d["value"] for d in descriptions if d.get("lang") == "en"), "")

    metrics = cve_obj.get("metrics", {})
    cvss_score = None
    cvss_vector = None
    for key in ["cvssMetricV40", "cvssMetricV31", "cvssMetricV30", "cvssMetricV2"]:
        if key in metrics and metrics[key]:
            m = metrics[key][0].get("cvssData", {})
            cvss_score = m.get("baseScore")
            cvss_vector = m.get("vectorString")
            break

    cpes = []
    for cfg in cve_obj.get("configurations", []):
        for node in cfg.get("nodes", []):
            for cpe_match in node.get("cpeMatch", []):
                if cpe_match.get("criteria"):
                    cpes.append(cpe_match["criteria"])

    return {
        "cve": cve_id,
        "description_en": en_desc,
        "cvss_score": cvss_score,
        "cvss_vector": cvss_vector,
        "cpes": cpes,
        "published": cve_obj.get("published"),
        "last_modified": cve_obj.get("lastModified"),
    }


def load_nvd_dump(dump_dir: Path) -> dict:
    """Читает все JSON-файлы NVD из папки. Возвращает {cve_id: {...}}."""
    print(f"\n=== Шаг 2b: Загрузка NVD из {dump_dir} ===")

    if not dump_dir.exists():
        print(f"[!] Папка {dump_dir} не найдена.")
        return {}

    files = (
        sorted(dump_dir.glob("*.json")) +
        sorted(dump_dir.glob("*.json.gz")) +
        sorted(dump_dir.glob("*.json.xz"))
    )
    if not files:
        print("[!] В папке нет JSON-файлов.")
        return {}

    print(f"Найдено файлов: {len(files)}")
    nvd_index = {}

    for path in files:
        print(f"  Читаю {path.name} ...")
        try:
            data = read_json_auto(path)
        except Exception as e:
            print(f"    [Ошибка] {e}")
            continue

        vulns = data.get("vulnerabilities", [])
        for item in tqdm(vulns, desc=f"  {path.stem}", leave=False):
            parsed = parse_nvd_item(item)
            if parsed:
                nvd_index[parsed["cve"]] = parsed

    print(f"Всего CVE в NVD-индексе: {len(nvd_index)}")
    return nvd_index


# =============================================================
# 6. ИЗВЛЕЧЕНИЕ CVE ИЗ FSTEC
# =============================================================

def extract_cves_from_field(field) -> list:
    if not field or str(field).strip() == "":
        return []
    parts = re.split(r"[,\s;]+", str(field).strip())
    return [p.strip() for p in parts if CVE_PATTERN.match(p.strip())]


# =============================================================
# 7. ПОСТРОЕНИЕ ГРАФА
# =============================================================

def build_graph(df_fstec: pd.DataFrame, nvd_index: dict):
    print("\n=== Шаг 3: Построение графа BDU ↔ CVE ===")

    bdu_to_cve = {}
    cve_to_bdu = defaultdict(lambda: {
        "bdu_list": [],
        "nvd_description_en": "",
        "nvd_cvss_score": None,
        "nvd_cvss_vector": "",
        "nvd_cpes": [],
        "nvd_published": None,
        "nvd_last_modified": None,
    })

    for _, row in tqdm(df_fstec.iterrows(), total=len(df_fstec), desc="Обработка FSTEC"):
        bdu_id = str(row.get("id", "")).strip()
        if not bdu_id:
            continue

        cve_list = extract_cves_from_field(row.get("cve_id", ""))

        bdu_to_cve[bdu_id] = {
            "title": str(row.get("title", "")).strip(),
            "description": str(row.get("description", "")).strip(),
            "cve_list": cve_list,
            "nvd_available": any(cve in nvd_index for cve in cve_list),
            "cvss_v2_0_fstec": (
                float(row["cvss_v2_0"])
                if str(row.get("cvss_v2_0", "")).strip() not in ("", "nan")
                else None
            ),
            "source": "fstec",
        }

        for cve in cve_list:
            entry = cve_to_bdu[cve]
            if bdu_id not in entry["bdu_list"]:
                entry["bdu_list"].append(bdu_id)

            if cve in nvd_index:
                nvd = nvd_index[cve]
                if not entry["nvd_description_en"]:
                    entry["nvd_description_en"] = nvd["description_en"]
                    entry["nvd_cvss_score"] = nvd["cvss_score"]
                    entry["nvd_cvss_vector"] = nvd["cvss_vector"]
                    entry["nvd_cpes"] = nvd["cpes"]
                    entry["nvd_published"] = nvd["published"]
                    entry["nvd_last_modified"] = nvd["last_modified"]

    return bdu_to_cve, dict(cve_to_bdu)


# =============================================================
# 8. UNMATCHED
# =============================================================

def find_unmatched(bdu_to_cve: dict, cve_to_bdu: dict, nvd_index: dict):
    print("\n=== Шаг 4: Поиск несопоставленных записей ===")

    bdu_without_cve = [
        bdu for bdu, data in bdu_to_cve.items() if not data["cve_list"]
    ]

    known_cves = set(cve_to_bdu.keys())
    all_nvd_cves = set(nvd_index.keys())
    cve_without_bdu = sorted(all_nvd_cves - known_cves)

    print(f"BDU без CVE:  {len(bdu_without_cve)}")
    print(f"CVE без BDU:  {len(cve_without_bdu)}")

    return bdu_without_cve, cve_without_bdu


# =============================================================
# 9. СОХРАНЕНИЕ JSON
# =============================================================

def save_json(bdu_to_cve, cve_to_bdu, bdu_wo, cve_wo, path: Path):
    print("\n=== Шаг 5a: Сохранение JSON ===")

    graph = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_bdu": len(bdu_to_cve),
            "total_cve": len(cve_to_bdu),
            "total_links": sum(len(v["cve_list"]) for v in bdu_to_cve.values()),
            "bdu_without_cve_count": len(bdu_wo),
            "cve_without_bdu_count": len(cve_wo),
        },
        "bdu_to_cve": bdu_to_cve,
        "cve_to_bdu": cve_to_bdu,
        "unmatched": {
            "bdu_without_cve": bdu_wo,
            "cve_without_bdu": cve_wo,
        },
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)

    size_mb = path.stat().st_size / 1024 / 1024
    print(f"[OK] JSON сохранён: {path}")
    print(f"     Размер:       {size_mb:.2f} МБ")


# =============================================================
# 10. СОХРАНЕНИЕ PARQUET
# =============================================================

def save_parquet(bdu_to_cve: dict, cve_to_bdu: dict, out_dir: Path):
    """
    Сохраняет две таблицы:
    - bdu_to_cve.parquet : строки = BDU
    - cve_to_bdu.parquet : строки = CVE
    """
    print("\n=== Шаг 5b: Сохранение Parquet ===")
    out_dir.mkdir(exist_ok=True)

    # --- Таблица 1: bdu_to_cve ---
    bdu_rows = []
    for bdu_id, data in bdu_to_cve.items():
        bdu_rows.append({
            "bdu_id": bdu_id,
            "title": data["title"],
            "description": data["description"],
            "cve_list": data["cve_list"],          # нативный список
            "nvd_available": data["nvd_available"],
            "cvss_v2_0_fstec": data["cvss_v2_0_fstec"],
            "source": data["source"],
        })

    df_bdu = pd.DataFrame(bdu_rows)
    bdu_path = out_dir / "bdu_to_cve.parquet"
    df_bdu.to_parquet(bdu_path, index=False, engine="pyarrow")
    print(f"[OK] {bdu_path}  ({len(df_bdu)} строк)")

    # --- Таблица 2: cve_to_bdu ---
    cve_rows = []
    for cve_id, data in cve_to_bdu.items():
        cve_rows.append({
            "cve_id": cve_id,
            "bdu_list": data["bdu_list"],          # нативный список
            "nvd_description_en": data["nvd_description_en"],
            "nvd_cvss_score": data["nvd_cvss_score"],
            "nvd_cvss_vector": data["nvd_cvss_vector"],
            "nvd_cpes": data["nvd_cpes"],          # нативный список
            "nvd_published": data["nvd_published"],
            "nvd_last_modified": data["nvd_last_modified"],
        })

    df_cve = pd.DataFrame(cve_rows)
    cve_path = out_dir / "cve_to_bdu.parquet"
    df_cve.to_parquet(cve_path, index=False, engine="pyarrow")
    print(f"[OK] {cve_path}  ({len(df_cve)} строк)")


# =============================================================
# 11. MAIN
# =============================================================

def main():
    df_fstec = load_fstec()

    # Приоритет: локальный дамп → API
    nvd_index = {}
    if NVD_DUMP_DIR.exists() and any(NVD_DUMP_DIR.glob("*.json*")):
        nvd_index = load_nvd_dump(NVD_DUMP_DIR)
    elif USE_API_IF_NO_DUMP:
        print("[i] Локальный дамп не найден, использую API NVD.")
        if not NVD_API_KEY:
            print("[i] API-ключ не задан. Лимит: 5 запросов / 30 сек [citation:6].")
            print("    Рекомендуется получить бесплатный ключ: https://nvd.nist.gov/developers/request-an-api-key")
            answer = input("    Продолжить без ключа? Это займёт много времени [y/N]: ").strip().lower()
            if answer != "y":
                print("Выход. Задайте NVD_API_KEY или положите дамп в ./nvd_dumps.")
                return
        nvd_index = fetch_nvd_via_api()
    else:
        print("[!] Нет ни дампа, ни разрешения на API. Продолжаю только с FSTEC.")

    bdu_to_cve, cve_to_bdu = build_graph(df_fstec, nvd_index)
    bdu_wo, cve_wo = find_unmatched(bdu_to_cve, cve_to_bdu, nvd_index)

    print(f"\nИтог:")
    print(f"  Уникальных BDU: {len(bdu_to_cve)}")
    print(f"  Уникальных CVE: {len(cve_to_bdu)}")
    print(f"  Всего связей:   {sum(len(v['cve_list']) for v in bdu_to_cve.values())}")

    save_json(bdu_to_cve, cve_to_bdu, bdu_wo, cve_wo, OUTPUT_JSON)
    save_parquet(bdu_to_cve, cve_to_bdu, OUTPUT_PARQUET_DIR)


if __name__ == "__main__":
    main()