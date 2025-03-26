import csv
from typing import List, Tuple
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from itertools import product
import pandas as pd
import argparse
import os


def load_tokenized_documents(csv_file_path: str) -> List[TaggedDocument]:
    documents = []
    with open(csv_file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            if len(row) < 3:
                continue
            tokens = row[1:]
            documents.append(TaggedDocument(words=tokens, tags=[str(idx)]))
    return documents


def train_doc2vec_model(
    documents: List[TaggedDocument],
    vector_size: int,
    epochs: int,
    min_count: int,
    window: int,
) -> Doc2Vec:
    model = Doc2Vec(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        epochs=epochs,
    )
    model.build_vocab(documents)
    model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
    return model


def evaluate_model(
    model: Doc2Vec, documents: List[TaggedDocument]
) -> Tuple[float, float]:
    correct_at_1 = 0
    correct_at_2 = 0
    total = len(documents)
    for i, doc in enumerate(documents):
        if i % 100 == 0 and i <= 1000:
            print(i)
        inferred_vector = model.infer_vector(doc.words)
        sims = model.dv.most_similar([inferred_vector], topn=2)
        top_tags = [tag for tag, _ in sims]
        if doc.tags[0] == top_tags[0]:
            correct_at_1 += 1
        if doc.tags[0] in top_tags:
            correct_at_2 += 1
    percent_at_1 = correct_at_1 / total * 100
    percent_at_2 = correct_at_2 / total * 100
    return percent_at_1, percent_at_2


def grid_search(
    documents: List[TaggedDocument], param_grid: dict, save_dir: str = "models"
) -> pd.DataFrame:
    results = []
    os.makedirs(save_dir, exist_ok=True)
    all_combinations = list(product(*param_grid.values()))
    for combo in all_combinations:
        params = dict(zip(param_grid.keys(), combo))
        print(f"\n🔍 Training with params: {params}")
        model = train_doc2vec_model(
            documents,
            vector_size=params["vector_size"],
            epochs=params["epochs"],
            min_count=params["min_count"],
            window=params["window"],
        )
        sim1, sim2 = evaluate_model(model, documents)
        model_name = f"d2v_v{params['vector_size']}_e{params['epochs']}_m{params['min_count']}_w{params['window']}.model"
        model_path = os.path.join(save_dir, model_name)
        # ✅ Save model only if data is sufficient and sim@2 > 80%
        if len(documents) >= 1000 and sim2 >= 80.0:
            print(f"Self Similarity {sim1:.3f}%")
            print(f"Second Self Similarity {sim2:.3f}%")
            model.save(model_path)
            print(f"💾 Model saved to {model_path}")
            saved = True
        else:
            print(f"❌ Model not saved (Second Self-Similarity = {sim2:.2f}%)")
            saved = False

        results.append(
            {
                **params,
                "self_similarity@1": sim1,
                "self_similarity@2": sim2,
                "model_saved": saved,
                "model_path": model_path if saved else "",
            }
        )
    return pd.DataFrame(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Doc2Vec model with or without Grid Search"
    )
    parser.add_argument(
        "--csv", type=str, default="sample_tokenized_data.csv", help="CSV path"
    )
    parser.add_argument(
        "--save_csv",
        type=str,
        default="grid_search_results.csv",
        help="Result CSV path",
    )
    parser.add_argument(
        "--grid_search", action="store_true", help="Enable grid search mode"
    )
    parser.add_argument(
        "--grid_size",
        type=str,
        choices=["small", "full"],
        default="small",
        help="Grid search parameter size, small: 1000, full: 60000",
    )

    parser.add_argument(
        "--save_model",
        action="store_true",
        help="Whether to save the model if similarity condition is met",
    )

    # default training parameters (for single-model training)
    parser.add_argument("--vector_size", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--min_count", type=int, default=2)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--save_path", type=str, default="models/d2v_custom.model")

    args = parser.parse_args()

    print("📥 Loading documents...")
    docs = load_tokenized_documents(args.csv)
    print("Titles Ready")
    print("Tagged Documents Ready")

    if args.grid_search:
        print("🚀 Running Grid Search...")
        if args.grid_size == "small":
            param_grid = {
                "vector_size": [50, 100, 150],
                "epochs": [20, 30, 50, 100],
                "min_count": [1, 2],
                "window": [3, 5],
            }
        elif args.grid_size == "full":
            param_grid = {
                "vector_size": [50, 100, 150],
                "epochs": [20, 30, 40],
                "min_count": [1],
                "window": [3, 5],
            }
        result_df = grid_search(docs, param_grid)
        result_df.to_csv(args.save_csv, index=False)
        print(f"✅ Grid search completed. Results saved to {args.save_csv}")
    else:
        print("Start Training")
        model = train_doc2vec_model(
            docs,
            vector_size=args.vector_size,
            epochs=args.epochs,
            min_count=args.min_count,
            window=args.window,
        )

        print("Test Similarity")
        sim1, sim2 = evaluate_model(model, docs)

        print(f"Self Similarity {sim1:.3f}%")
        print(f"Second Self Similarity {sim2:.3f}%")

        if args.save_model:
            if len(docs) >= 1000 and sim2 >= 80.0:
                model.save(args.save_path)
                print(f"💾 Model saved to {args.save_path}")
            else:
                print(
                    "❌ Model not saved (Second Self Similarity < 80% or insufficient data)"
                )
        else:
            print("🛑 Model not saved (save_model flag not set)")
