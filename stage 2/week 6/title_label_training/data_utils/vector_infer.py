from typing import Callable, List

import numpy as np
from gensim.models.doc2vec import Doc2Vec


def tokens_to_vectors_mp(
    model: Doc2Vec, tokenized_texts: List[List[str]], workers: int = 4
) -> np.ndarray:
    from multiprocessing import Pool

    with Pool(workers) as pool:
        vectors = pool.map(model.infer_vector, tokenized_texts)
    return np.array(vectors)


def infer_with_doc2vec(model: Doc2Vec, tokens: List[str]) -> np.ndarray:
    return model.infer_vector(tokens)


def get_infer_vector_fn(
    method: str = 'doc2vec',
) -> Callable[[any, List[str]], np.ndarray]:
    """
    回傳對應的 infer_vector function

    Args:
        method (str): 選擇 'doc2vec'

    Returns:
        Callable: (model, tokens) -> np.ndarray
    """
    method = method.lower()

    if method == 'doc2vec':
        return infer_with_doc2vec
    else:
        raise ValueError(f'Unsupported method: {method}')
