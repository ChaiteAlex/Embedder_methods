"""Очистка датасета MITRE перед переводом.

Примеры запуска:
    # Один файл, вывод в папку cleaned/
    python clean_dataset.py data/train.jsonl -o cleaned/

    # Несколько файлов, вывод в ту же папку
    python clean_dataset.py data/train.jsonl data/test.jsonl data/val.jsonl -o cleaned/

    # Все .jsonl из папки
    python clean_dataset.py data/ -o cleaned/

    # Один файл с явным именем выхода
    python clean_dataset.py data/train.jsonl -o cleaned/train_cleaned.jsonl
"""

import argparse
import json
import re
import sys
from pathlib import Path


CITATION = re.compile(r'\(Citation:[^)]*\)')
MD_LINK  = re.compile(r'\[([^\]]+)\]\((?:https?://)?[^)\s]*\)')
URL      = re.compile(r'(?:https?|hxxps?)://[^\s)\]>]+', re.I)
CODE_TAG = re.compile(r'</?code>', re.I)
BR_TAG   = re.compile(r'<br\s*/?>', re.I)
REF_NUM  = re.compile(r'\[\d{1,3}\]')

TRANS = str.maketrans({
    '’': "'", '‘': "'", '´': "'", '“': '"', '”': '"',
    '–': '-', '—': '-', '…': '...', '\xa0': ' ', '\u202f': ' ',
    '\u200b': '', '\xad': '', '\uf0b7': '', '•': '', '▼': '',
    '®': '', '™': '', '©': '',
})


def clean(text: str) -> str:
    if not isinstance(text, str):
        return ''
    t = CITATION.sub('', text)
    t = MD_LINK.sub(r'\1', t)
    t = URL.sub('', t)
    t = BR_TAG.sub(' ', t)
    t = CODE_TAG.sub('', t)
    t = REF_NUM.sub('', t)
    t = t.translate(TRANS)
    t = re.sub(r'\s+', ' ', t)
    t = re.sub(r'\s+([.,;:!?])', r'\1', t)
    t = re.sub(r'\(\s*\)', '', t)
    return t.strip()


def process_file(src: Path, dst: Path) -> tuple[int, int]:
    """Возвращает (оставлено, удалено)."""
    dst.parent.mkdir(parents=True, exist_ok=True)

    kept = dropped = 0
    with open(src, encoding='utf-8') as f, open(dst, 'w', encoding='utf-8') as out:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [!] {src.name}: пропущена битая строка ({e})", file=sys.stderr)
                dropped += 1
                continue

            for k in ('event_text', 'technique_description'):
                row[k] = clean(row.get(k))

            if not row['event_text'] or not row['technique_description']:
                dropped += 1
                continue

            out.write(json.dumps(row, ensure_ascii=False) + '\n')
            kept += 1

    return kept, dropped


def collect_inputs(inputs: list[str]) -> list[Path]:
    """Разворачивает папки в список .jsonl-файлов."""
    files: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            files.extend(sorted(p.glob("*.jsonl")))
        elif p.is_file():
            files.append(p)
        else:
            print(f"[!] Не найден: {item}", file=sys.stderr)
    return files


def resolve_output(src: Path, output: Path | None, suffix: str) -> Path:
    """
    Определяет путь выхода:
      - output=None                → рядом с исходным: <name>_cleaned.jsonl
      - output=папка               → <output>/<name>_cleaned.jsonl
      - output=файл (для 1 входа)  → output как есть
      - output=файл (для N входов) → <output.stem>_<src.stem><suffix>
    """
    if output is None:
        return src.with_name(src.stem + suffix + src.suffix)

    if output.is_dir() or not output.suffix:
        return output / (src.stem + suffix + src.suffix)

    # output — файл
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Очистка MITRE JSONL: удаление цитат, ссылок, HTML-тегов.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("inputs", nargs="+",
                        help="Файлы .jsonl или папки с ними")
    parser.add_argument("-o", "--output", default=None,
                        help="Папка или файл для результата. "
                             "По умолчанию: <name>_cleaned.jsonl рядом с исходником")
    parser.add_argument("--suffix", default="_cleaned",
                        help="Суффикс к имени файла (по умолчанию: _cleaned)")
    args = parser.parse_args()

    files = collect_inputs(args.inputs)
    if not files:
        print("[!] Нет входных файлов.", file=sys.stderr)
        sys.exit(1)

    output = Path(args.output) if args.output else None
    # Если на входе один файл и -o указывает на файл — пишем прямо туда
    single_file_output = (
        output is not None and len(files) == 1
        and output.suffix.lower() == ".jsonl"
    )

    total_kept = total_dropped = 0
    for src in files:
        dst = output if single_file_output else resolve_output(src, output, args.suffix)
        print(f"→ {src}  →  {dst}")
        kept, dropped = process_file(src, dst)
        total_kept += kept
        total_dropped += dropped
        print(f"  оставлено: {kept}, удалено пустых: {dropped}")

    print()
    print(f"Всего: оставлено {total_kept}, удалено {total_dropped}")


if __name__ == "__main__":
    main()