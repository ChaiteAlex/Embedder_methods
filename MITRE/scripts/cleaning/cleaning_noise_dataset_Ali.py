"""Очистка датасета Mitra перед переводом.
Запуск: python clean_dataset.py train_ml_clean.jsonl train_ml_cleaned.jsonl
"""
import json, re, sys

CITATION = re.compile(r'\(Citation:[^)]*\)')                 # (Citation: Источник)
MD_LINK  = re.compile(r'\[([^\]]+)\]\((?:https?://)?[^)\s]*\)')  # [текст](ссылка) -> текст
URL      = re.compile(r'(?:https?|hxxps?)://[^\s)\]>]+', re.I)   # ссылки со схемой
CODE_TAG = re.compile(r'</?code>', re.I)                     # <code> и </code>
BR_TAG   = re.compile(r'<br\s*/?>', re.I)
REF_NUM  = re.compile(r'\[\d{1,3}\]')                        # сноски вида [12]

# типографские символы -> обычные
TRANS = str.maketrans({'’':"'", '‘':"'", '´':"'", '“':'"', '”':'"', '–':'-', '—':'-',
                       '…':'...', '\xa0':' ', '\u202f':' ', '\u200b':'', '\xad':'',
                       '\uf0b7':'', '•':'', '▼':'', '®':'', '™':'', '©':''})

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
    t = re.sub(r'\s+', ' ', t)                 # переносы строк, табы, двойные пробелы
    t = re.sub(r'\s+([.,;:!?])', r'\1', t)     # пробел перед знаком препинания
    t = re.sub(r'\(\s*\)', '', t)              # пустые скобки после удаления
    return t.strip()

def main(src, dst):
    kept = dropped = 0
    with open(src, encoding='utf-8') as f, open(dst, 'w', encoding='utf-8') as out:
        for line in f:
            row = json.loads(line)
            for k in ('event_text', 'technique_description'):
                row[k] = clean(row.get(k))
            if not row['event_text'] or not row['technique_description']:
                dropped += 1
                continue
            out.write(json.dumps(row, ensure_ascii=False) + '\n')
            kept += 1
    print(f'{src}: оставлено {kept}, удалено пустых {dropped}')

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
