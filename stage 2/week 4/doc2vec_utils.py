import csv
import os
from typing import List, Tuple, Dict
from itertools import product
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


def load_tokenized_documents(csv_file_path: str) -> List[TaggedDocument]:
    documents = []
    with open(csv_file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            if len(row) < 2:
                continue
            tokens = row[1:]
            documents.append(TaggedDocument(words=tokens, tags=[str(idx)]))
    return documents


def train_doc2vec_model(
    documents: List[TaggedDocument],
    vector_size: int = 100,
    epochs: int = 100,
    min_count: int = 2,
    window: int = 5,
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


def grid_search_doc2vec(
    documents: List[TaggedDocument],
    param_grid: Dict[str, List],
    save_dir: str = 'models',
) -> pd.DataFrame:
    results = []
    os.makedirs(save_dir, exist_ok=True)
    all_combinations = list(product(*param_grid.values()))
    for combo in all_combinations:
        params = dict(zip(param_grid.keys(), combo))
        print(f'\n🔍 Training with params: {params}')
        model = train_doc2vec_model(
            documents,
            vector_size=params['vector_size'],
            epochs=params['epochs'],
            min_count=params['min_count'],
            window=params['window'],
        )
        sim1, sim2 = evaluate_model(model, documents)
        model_name = f'd2v_v{params["vector_size"]}_e{params["epochs"]}_m{params["min_count"]}_w{params["window"]}.model'
        model_path = os.path.join(save_dir, model_name)
        saved = False
        if len(documents) >= 1000 and sim2 >= 80.0:
            print(f'Self-Similarity@1: {sim1:.2f}%  |  Self-Similarity@2: {sim2:.2f}%')
            model.save(model_path)
            print(f'💾 Saved model to {model_path}')
            saved = True
        results.append(
            {
                **params,
                'self_similarity@1': sim1,
                'self_similarity@2': sim2,
                'model_saved': saved,
                'model_path': model_path if saved else '',
            }
        )
    return pd.DataFrame(results)
