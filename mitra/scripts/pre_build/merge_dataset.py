"""Добавляет к labels все поля соответствующих техник из трёх Excel-файлов."""

import argparse
import ast
import csv
import hashlib
import json
import posixpath
import re
import shutil
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE_DIR = HERE / "sources"
OUTPUT_DIR = HERE / "dataset"
REPOSITORY = "tumeteor/Security-TTP-Mapping"
WORKBOOKS = (
    "enterprise-attack-v19.2.xlsx",
    "ics-attack-v19.2.xlsx",
    "mobile-attack-v19.2.xlsx",
)
NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
LABEL_PATTERN = re.compile(r"T\d{4}(?:\.\d{3})?\Z")


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Сначала пишем временный файл, чтобы не оставить недокачанный исходник.
    temporary = path.with_suffix(path.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "dataset-label-merge/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        with temporary.open("wb") as stream:
            shutil.copyfileobj(response, stream)
    temporary.replace(path)


def source_files(allow_download=False, allow_missing=False):
    metadata_path = SOURCE_DIR / "hf_metadata.json"
    if not metadata_path.exists():
        if not allow_download:
            raise FileNotFoundError("Нет исходных данных. Сначала запустите с --download.")
        download(f"https://huggingface.co/api/datasets/{REPOSITORY}", metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    revision = metadata["sha"]
    files = []
    # Берём все части с размеченными текстами; справочник ttp-desc-mappings не примешиваем.
    for entry in sorted(metadata["siblings"], key=lambda item: item["rfilename"]):
        name = entry["rfilename"]
        match = re.fullmatch(r"datasets/[^/]+/[^/]+_(train|dev|test)\.tsv", name)
        if not match:
            continue
        split = "validation" if match[1] == "dev" else match[1]
        path = SOURCE_DIR / name
        if not path.exists():
            if not allow_download:
                if allow_missing:
                    continue
                raise FileNotFoundError(f"Нет {path.name}. Запустите с --download.")
            # Фиксируем версию, чтобы повторный запуск использовал те же данные.
            download(f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{revision}/{name}", path)
        files.append((split, name, path))
    if not files:
        raise ValueError("В репозитории не найдены train/dev/test TSV.")
    return revision, files


def excel_rows(path, sheet_name="techniques"):
    """Читает нужный лист XLSX без установки дополнительных библиотек."""
    # XLSX — это ZIP с XML-файлами. Здесь нужны только значения ячеек.
    with zipfile.ZipFile(path) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            strings = [
                "".join(part.text or "" for part in item.findall(".//s:t", NS))
                for item in root
            ]

        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        sheets = workbook.find("s:sheets", NS)
        sheet = next((item for item in sheets if item.get("name") == sheet_name), None)
        if sheet is None:
            raise ValueError(f"В {path.name} нет листа {sheet_name}")
        relationship_id = sheet.get(f"{{{REL_NS}}}id")
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        target = next(item.get("Target") for item in relationships if item.get("Id") == relationship_id)
        sheet_path = target.lstrip("/") if target.startswith("/") else posixpath.normpath(posixpath.join("xl", target))

        def read_cell(cell):
            kind = cell.get("t")
            if kind == "inlineStr":
                return "".join(item.text or "" for item in cell.findall(".//s:t", NS))
            value = cell.find("s:v", NS)
            if value is None or value.text is None:
                return None
            text = value.text
            if kind == "s":
                return strings[int(text)]
            if kind == "b":
                return text == "1"
            if kind in ("str", "e", "d"):
                return text
            number = float(text)
            return int(number) if number.is_integer() else number

        with archive.open(sheet_path) as stream:
            headers = None
            for _, element in ET.iterparse(stream, events=("end",)):
                if element.tag != f"{{{NS['s']}}}row":
                    continue
                # Ячейки могут быть пропущены: связываем их по буквам столбца.
                cells = {re.sub(r"\d", "", cell.get("r")): read_cell(cell) for cell in element.findall("s:c", NS)}
                if headers is None:
                    headers = {column: name for column, name in cells.items() if name is not None}
                    if "ID" not in headers.values():
                        raise ValueError(f"В {path.name} не найден столбец ID")
                    if len(set(headers.values())) != len(headers):
                        raise ValueError(f"Повторяющиеся заголовки в {path.name}")
                elif any(value is not None for value in cells.values()):
                    yield {name: cells.get(column) for column, name in headers.items()}
                element.clear()


def build_index():
    index = {}
    workbooks = []
    for name in WORKBOOKS:
        path = HERE.parent / name
        count = 0
        for row in excel_rows(path):
            label = row["ID"]
            if not isinstance(label, str) or not LABEL_PATTERN.fullmatch(label):
                raise ValueError(f"Некорректный ID в {name}: {label!r}")
            if label in index and index[label] != row:
                raise ValueError(f"Для {label} найдены разные данные. Нельзя выбрать строку автоматически.")
            index[label] = row
            count += 1
        workbooks.append({"file": name, "sheet": "techniques", "rows": count, "sha256": file_hash(path)})
    return index, workbooks


def parse_labels(value):
    # В TSV список записан текстом: ['T1057', 'T1113']. eval здесь не нужен.
    labels = ast.literal_eval(value)
    if not isinstance(labels, list) or any(
        not isinstance(label, str) or not LABEL_PATTERN.fullmatch(label) for label in labels
    ):
        raise ValueError(f"Неожиданный формат labels: {value!r}")
    return labels


def enrich(row, index):
    if "attack_details" in row:
        raise ValueError("Столбец attack_details уже существует")
    # Сохраняем порядок меток и их повторы. Для отсутствующих ID оставляем null.
    details = [{"label": label, "data": index.get(label)} for label in parse_labels(row["labels"])]
    return {**row, "attack_details": details}


def merge(index, files, revision, workbooks):
    OUTPUT_DIR.mkdir(exist_ok=True)
    metadata = json.loads((SOURCE_DIR / "hf_metadata.json").read_text(encoding="utf-8-sig"))
    expected = [entry["rfilename"] for entry in metadata["siblings"]
                if re.fullmatch(r"datasets/[^/]+/[^/]+_(train|dev|test)\.tsv", entry["rfilename"])]
    present = {name for _, name, _ in files}
    missing_files = sorted(set(expected) - present)
    report = {
        "dataset": REPOSITORY,
        "revision": revision,
        "added_column": "attack_details",
        "complete": not missing_files,
        "missing_source_files": missing_files,
        "workbooks": workbooks,
        "technique_count": len(index),
        "splits": {},
    }
    all_missing = Counter()
    for split in ("train", "validation", "test"):
        output = OUTPUT_DIR / f"{split}.jsonl"
        temporary = output.with_suffix(".jsonl.part")
        stats = {"rows": 0, "label_occurrences": 0, "matched_occurrences": 0,
                 "rows_with_missing_labels": 0, "source_files": []}
        suffix = "dev" if split == "validation" else split
        stats["missing_source_files"] = [name for name in missing_files if name.endswith(f"_{suffix}.tsv")]
        stats["complete"] = not stats["missing_source_files"]
        missing = Counter()
        columns = None
        with temporary.open("w", encoding="utf-8", newline="\n") as destination:
            for file_split, name, path in files:
                if file_split != split:
                    continue
                start_row = stats["rows"]
                with path.open(encoding="utf-8-sig", newline="") as stream:
                    reader = csv.DictReader(stream, delimiter="\t")
                    if "labels" not in (reader.fieldnames or []):
                        raise ValueError(f"В {name} нет столбца labels")
                    if columns is None:
                        columns = reader.fieldnames
                    elif columns != reader.fieldnames:
                        raise ValueError(f"В {name} другой набор столбцов")
                    for row in reader:
                        if None in row or any(value is None for value in row.values()):
                            raise ValueError(f"Повреждённая строка в {name}, строка {reader.line_num}")
                        enriched = enrich(row, index)
                        details = enriched["attack_details"]
                        absent = [item["label"] for item in details if item["data"] is None]
                        missing.update(absent)
                        stats["rows"] += 1
                        stats["label_occurrences"] += len(details)
                        stats["matched_occurrences"] += len(details) - len(absent)
                        stats["rows_with_missing_labels"] += bool(absent)
                        destination.write(json.dumps(enriched, ensure_ascii=False) + "\n")
                stats["source_files"].append({"file": name, "rows": stats["rows"] - start_row,
                                             "output_start_row_zero_based": start_row, "sha256": file_hash(path)})
        temporary.replace(output)
        stats["original_columns"] = columns
        stats["missing_labels"] = dict(sorted(missing.items()))
        stats["output"] = str(output.relative_to(HERE))
        stats["output_sha256"] = file_hash(output)
        report["splits"][split] = stats
        all_missing.update(missing)
        status = "complete" if stats["complete"] else "INCOMPLETE"
        print(f"{split}: {stats['rows']} rows, {stats['matched_occurrences']}/{stats['label_occurrences']} labels matched; {status}")
    report["missing_labels"] = dict(sorted(all_missing.items()))
    report["total_rows"] = sum(stats["rows"] for stats in report["splits"].values())
    (HERE / "merge_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="скачать недостающие исходные TSV с Hugging Face")
    parser.add_argument("--download-only", action="store_true", help="только скачать исходники")
    parser.add_argument("--allow-missing", action="store_true", help="явно разрешить неполный набор и перечислить пропуски в отчёте")
    args = parser.parse_args()
    revision, files = source_files(args.download or args.download_only, allow_missing=args.allow_missing)
    if args.download_only:
        print(f"Downloaded {len(files)} source files at {revision}")
        return
    index, workbooks = build_index()
    report = merge(index, files, revision, workbooks)
    print(f"Total: {report['total_rows']} rows; {len(report['missing_labels'])} unmatched unique labels")


if __name__ == "__main__":
    main()
