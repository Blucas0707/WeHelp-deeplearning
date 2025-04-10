from typing import List

import numpy as np
from gensim.models.doc2vec import Doc2Vec


def tokens_to_vectors_mp(
    model: Doc2Vec, tokenized_texts: List[List[str]], workers: int = 4
) -> np.ndarray:
    from multiprocessing import Pool

    with Pool(workers) as pool:
        vectors = pool.map(model.infer_vector, tokenized_texts)
    return np.array(vectors)
