import multiprocessing as mp
from typing import List, Tuple, Union

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from tqdm import tqdm


def _evaluate_worker(
    doc_batch: List[TaggedDocument],
    model: Doc2Vec,
    c1: mp.Value,
    c2: mp.Value,
) -> None:
    local_1 = 0
    local_2 = 0
    for doc in doc_batch:
        inferred_vector = model.infer_vector(doc.words)
        sims = model.dv.most_similar([inferred_vector], topn=2)
        top_tags = [tag for tag, _ in sims]
        if doc.tags[0] == top_tags[0]:
            local_1 += 1
        if doc.tags[0] in top_tags:
            local_2 += 1
    with c1.get_lock(), c2.get_lock():
        c1.value += local_1
        c2.value += local_2


def evaluate_model(
    model: Doc2Vec,
    documents: List[TaggedDocument],
    sample_size: Union[int, None] = None,
    use_multiprocessing: bool = False,
    num_workers: Union[int, None] = None,
) -> Tuple[float, float]:
    if sample_size:
        documents = documents[:sample_size]

    total = len(documents)

    if use_multiprocessing:
        num_workers = num_workers or mp.cpu_count()
        print(f'🚀 Evaluating with multiprocessing ({num_workers} workers)...')
        correct_at_1 = mp.Value('i', 0)
        correct_at_2 = mp.Value('i', 0)

        chunk_size = (total + num_workers - 1) // num_workers
        processes = []
        for i in range(num_workers):
            chunk = documents[i * chunk_size : (i + 1) * chunk_size]
            p = mp.Process(
                target=_evaluate_worker, args=(chunk, model, correct_at_1, correct_at_2)
            )
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        percent_at_1 = correct_at_1.value / total * 100
        percent_at_2 = correct_at_2.value / total * 100
    else:
        print(f'🔍 Evaluating {total} documents without multiprocessing...')
        correct_1 = 0
        correct_2 = 0
        for doc in tqdm(documents, desc='🧠 Evaluating'):
            inferred_vector = model.infer_vector(doc.words)
            sims = model.dv.most_similar([inferred_vector], topn=2)
            top_tags = [tag for tag, _ in sims]
            if doc.tags[0] == top_tags[0]:
                correct_1 += 1
            if doc.tags[0] in top_tags:
                correct_2 += 1
        percent_at_1 = correct_1 / total * 100
        percent_at_2 = correct_2 / total * 100

    return percent_at_1, percent_at_2


def top_k_accuracy(logits, targets, k: int = 2) -> float:
    topk_preds = logits.topk(k=k, dim=1).indices
    correct = topk_preds.eq(targets.unsqueeze(1)).any(dim=1)
    return correct.float().mean().item()
