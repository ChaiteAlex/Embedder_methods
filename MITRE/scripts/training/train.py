"""
Скрипт обучения эмбеддера на скомпилированном плоском датасете.
"""
import argparse
import pandas as pd
import torch
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments
)
from sentence_transformers.evaluation import InformationRetrievalEvaluator

# ============================================================
# 1. ПРОСТОЙ DATASET (читает уже готовый текст)
# ============================================================

class ReadyPairDataset(torch.utils.data.Dataset):
    def __init__(self, parquet_path: str):
        print(f"Загрузка готового датасета: {parquet_path}")
        self.df = pd.read_parquet(parquet_path)
        print(f"   Загружено {len(self.df)} примеров.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # MultipleNegativesRankingLoss ожидает список текстов. 
        # Колонка 'label' игнорируется этим лоссом (он считает все в батче позитивами, а остальные - негативами)
        return InputExample(texts=[str(row['text_1']), str(row['text_2'])])


# ============================================================
# 2. EVALUATOR (для валидации на test_dataset)
# ============================================================

def create_evaluator(test_dataset_path: str):
    print("Подготовка Evaluator на основе test_dataset...")
    df = pd.read_parquet(test_dataset_path)
    
    queries = {}
    corpus = {}
    relevant_docs = {}

    for idx, row in df.iterrows():
        q_id = f"q_{idx}"
        d_id = f"d_{idx}"
        
        queries[q_id] = str(row['text_1'])
        corpus[d_id] = str(row['text_2'])
        
        if q_id not in relevant_docs:
            relevant_docs[q_id] = set()
        relevant_docs[q_id].add(d_id)

    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="cyber_test",
        show_progress_bar=False
    )


# ============================================================
# 3. ОБУЧЕНИЕ
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Обучение эмбеддера на плоском датасете")
    parser.add_argument("--train", required=True, help="Путь к train_dataset.parquet")
    parser.add_argument("--test", required=True, help="Путь к test_dataset.parquet")
    parser.add_argument("--output-dir", default="./models/cyber_embedder", help="Куда сохранить модель")
    parser.add_argument("--model-name", default="intfloat/multilingual-e5-small", help="Базовая модель")
    parser.add_argument("--batch-size", type=int, default=64, help="Размер батча (важно для InfoNCE!)")
    parser.add_argument("--epochs", type=int, default=3, help="Количество эпох")
    args = parser.parse_args()

    print(f"🚀 Обучение: {args.model_name} | Batch: {args.batch_size} | Epochs: {args.epochs}")

    # 1. Модель
    model = SentenceTransformer(args.model_name)

    # 2. Данные
    train_dataset = ReadyPairDataset(args.train)
    
    # 3. Лосс: MultipleNegativesRankingLoss (InfoNCE)
    # Идеален, когда в датасете только позитивные пары (label=1).
    # Он автоматически использует другие элементы батча как негативы.
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # 4. Валидация
    evaluator = create_evaluator(args.test)

    # 5. Аргументы Trainer
    training_args = SentenceTransformerTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_ratio=0.1,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="cyber_test_cosine_accuracy",
    )

    # 6. Запуск
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=train_loss,
        evaluator=evaluator,
    )

    print("🏋️ Начинаю обучение...")
    trainer.train()

    model.save_pretrained(args.output_dir)
    print(f"✅ Готово! Модель сохранена в: {args.output_dir}")

if __name__ == "__main__":
    main()