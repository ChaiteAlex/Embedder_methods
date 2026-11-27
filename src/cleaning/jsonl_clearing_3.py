import json
import os



def clean_jsonl_for_ml(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"{input_file} не найден.")
        return 0, 0

    processed = skipped = 0

    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)

                event_text = obj.get("event_text", "").strip()

                technique_desc = ""
                technique_desc = obj.get("technique_description", "").strip()

                label = obj.get("label","")
                parent_label = obj.get("parent_label","")
                tactics = obj.get("tactics", "")

                if event_text:
                    clean_obj = {
                        "event_text": event_text,
                        "technique_description": technique_desc,
                        "label": label,
                        "parent_label": parent_label,
                        "tactics": tactics
                    }
                    fout.write(json.dumps(clean_obj, ensure_ascii=False) + '\n')
                    processed += 1
                else:
                    skipped += 1

            except Exception as e:
                print(f"⚠️ Ошибка: {e}")
                skipped += 1

    return processed, skipped


if __name__ == "__main__":
    print("🧹 Чищу JSONL (без дублирования полей)...\n")
    for src, dst in [
        ("mitra\\final\\train.jsonl", "mitra\\final\\train_ml_clean.jsonl"),
        ("mitra\\final\\validation.jsonl", "mitra\\final\\validation_ml_clean.jsonl"),
        ("mitra\\final\\test.jsonl", "mitra\\final\\test_ml_clean.jsonl"),
    ]:
        p, s = clean_jsonl_for_ml(src, dst)
        print(f" {src} → {dst}: {p} сохранено, {s} пропущено")
    print("\nГотово!")