# Embedder_methods

## Description

### Разработка методологии поиска и сбора данных для обучения эмбеддера ИБ домена

Автоматизированный конвейер для сбора, очистки, нормализации и обучения эмбеддеров на данных кибербезопасности (MITRE ATT&CK, ФСТЭК (БДУ), NVD (CVE)). Проект формирует контрастивные датасеты пар и обучает модель, способную семантически связывать описания уязвимостей, техники атак и их коды на двух языках (RU/EN).

##  Architecture

```text
Embedder_methods/
├── README.md                          # Этот файл
├── requirements.txt                   # Зависимости проекта
├── .gitignore                         # Исключения для Git
│
├── docs/
│   └── data_sources_guideline.md      # Гайдлайн по 5 типам источников
│
├── src/
│   ├── analysis/
│   │   └── analyze_datasets.py        # Анализ и статистика датасетов
│   ├── cleaning/
│   │   ├── clean_mitre.py             # Очистка данных MITRE
│   │   ├── cleaning_noise_dataset.py  # Фильтрация шума
│   │   └── clear_json.py              # Базовая очистка JSON/JSONL
│   ├── collectors/
│   │   ├── fetch_fstec_nvd.py         # Сбор данных FSTEC + NVD
│   │   └── merge_dataset.py           # Объединение MITRE и TTP-Mapping
│   ├── dataset/
│   │   ├── parquet_restructure.py     # Нормализация в OAV-структуру
│   │   ├── build_pairs.py             # Генерация пар (позитивы)
│   │   ├── training_pairs_build.py    # Альтернативный генератор пар
│   │   └── compile_dataset.py         # Мультиязычная компиляция
│   └── training/
│       └── train.py                   # Обучение эмбеддера
│
└── data/
    ├── raw/                           # Исходные данные (CSV, JSON, JSONL)
    ├── intermediate/                  # Промежуточная обработка
    └── final/                         # Готовые к обучению данные
```

## Dependencies

Основные зависимости проекта (полный список в [`requirements.txt`](requirements.txt)):

| Категория | Пакеты |
|---|---|
| **ML и эмбеддеры** | `torch`, `sentence-transformers`, `transformers` |
| **Обработка данных** | `pandas`, `numpy`, `pyarrow`, `scikit-learn` |
| **Сбор и графы** | `requests`, `networkx`, `huggingface-hub` |
| **Утилиты** | `tqdm`, `scipy`, `Pillow` |

## Quick start

```bash
# 1. Установка зависимостей
pip install -r requirements.txt

# 2. Сбор данных
python src/collectors/fetch_fstec_nvd.py -o data/raw/fstec/

# 3. Очистка и валидация
python src/cleaning/clear_json.py -i data/raw/fstec/ -o data/intermediate/fstec/
python src/cleaning/validate_quality.py --input data/intermediate/fstec/texts.parquet

# 4. Генерация пар и обучение
python src/dataset/build_pairs.py -i data/intermediate/fstec/ -o data/final/pairs/
python src/training/train.py --train data/final/pairs/train_pairs.parquet --output-dir models/embedder_v1
```

## Results

    To be continued...


##  Credits

|№|Source data|Type|Structure relation|URL|License|
|---|---|---|---|---|---|
| 1 | MITRE ATT&CK STIX | CTI graph / STIX | relationship objects | https://github.com/mitre-attack/attack-stix-data | MITRE ATT&CK Terms of Use |
| 2 | Vulnerability-FSTEC | vulnerability records | BDU → CVE |https://huggingface.co/datasets/CIRCL/Vulnerability-FSTEC | CC BY 4.0 |
| 3 | Security-TTP-Mapping | security text + ATT&CK labels | text → [`Txxxx...`] | https://huggingface.co/datasets/tumeteor/Security-TTP-Mapping | CC BY 4.0 |
| 4 | ФСТЭК России| Semi-structured flat tabular text | Attribute-based logical mappings (Threat ↔ Source, Target, Impact) | https://bdu.fstec.ru | Public/Government data |