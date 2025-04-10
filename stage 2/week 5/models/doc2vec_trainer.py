import os
from itertools import product
from typing import List, Dict

import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from evaluation.evaluator import evaluate_model


def train_doc2vec_model(
    documents: List[TaggedDocument],
    vector_size: int = 100,
    epochs: int = 100,
    min_count: int = 2,
    window: int = 5,
    workers: int = 4,
) -> Doc2Vec:
    model = Doc2Vec(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
    )
    model.build_vocab(documents)
    model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
    return model


def grid_search_doc2vec(
    documents: List[TaggedDocument],
    param_grid: Dict[str, List],
    save_dir: str = 'models',
    eval_sample_size: int = 1000,
    use_multiprocessing: bool = True,
    num_workers: int = None,
    save_threshold: float = 80.0,
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
            workers=num_workers or os.cpu_count(),
        )

        sim1, sim2 = evaluate_model(
            model,
            documents,
            sample_size=eval_sample_size,
            use_multiprocessing=use_multiprocessing,
            num_workers=num_workers,
        )

        model_name = f'd2v_v{params["vector_size"]}_e{params["epochs"]}_m{params["min_count"]}_w{params["window"]}.model'
        model_path = os.path.join(save_dir, model_name)
        saved = False

        if len(documents) >= 1000 and sim2 >= save_threshold:
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
