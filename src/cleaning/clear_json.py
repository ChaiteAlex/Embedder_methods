import json


def create_bdu_cve_flat(input_file, output_file, include_ids=True):
    """
    Создаёт плоский JSONL-файл с парами BDU-CVE.
    
    :param input_file: путь к исходному JSON-файлу маппинга
    :param output_file: путь для сохранения JSONL
    :param include_ids: если True — сохраняет bdu_id и cve_id,
                        если False — оставляет только описания
    """
    print(f"Загрузка {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    # Рекурсивная очистка ключей и строк от пробелов по краям
    def clean(obj):
        if isinstance(obj, dict):
            return {k.strip(): clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean(x) for x in obj]
        if isinstance(obj, str):
            return obj.strip()
        return obj

    data = clean(raw)
    bdu_to_cve = data.get("bdu_to_cve", {})
    cve_to_bdu = data.get("cve_to_bdu", {})

    records = []
    bdu_without_cve = 0
    bdu_with_multi_cve = 0

    for bdu_id, bdu_data in bdu_to_cve.items():
        cve_list = bdu_data.get("cve_list", [])
        description_ru = bdu_data.get("description", "")

        if not cve_list:
            # BDU без связанного CVE
            bdu_without_cve += 1
            record = {
                "description_ru": description_ru,
                "description_en": ""
            }
            if include_ids:
                record["bdu_id"] = bdu_id
                record["cve_id"] = None
            records.append(record)
        else:
            if len(cve_list) > 1:
                bdu_with_multi_cve += 1

            for cve_id in cve_list:
                cve_data = cve_to_bdu.get(cve_id, {})
                description_en = cve_data.get("nvd_description_en", "")

                record = {
                    "description_ru": description_ru,
                    "description_en": description_en
                }
                if include_ids:
                    record["bdu_id"] = bdu_id
                    record["cve_id"] = cve_id
                records.append(record)

    # Сохраняем в JSONL
    print(f"Сохранение {len(records)} записей в {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    # Статистика
    mode = "С ID" if include_ids else "БЕЗ ID"
    print(f"\nРежим: {mode}")
    print(f"   Всего BDU в исходнике: {len(bdu_to_cve)}")
    print(f"   BDU без CVE: {bdu_without_cve}")
    print(f"   BDU с несколькими CVE (дублирование): {bdu_with_multi_cve}")
    print(f"   Итоговых записей: {len(records)}")


if __name__ == "__main__":
    # Пример 1: Сохраняем С идентификаторами (для аналитики, RAG)
    create_bdu_cve_flat(
        "bdu_cve_mapping.json",
        "bdu_cve_with_ids.jsonl",
        include_ids=True
    )

    # Пример 2: Сохраняем БЕЗ идентификаторов (для чистых ML-эмбеддингов)
    create_bdu_cve_flat(
        "bdu_cve_mapping.json",
        "bdu_cve_no_ids.jsonl",
        include_ids=False
    )

    print("\n Готово!")