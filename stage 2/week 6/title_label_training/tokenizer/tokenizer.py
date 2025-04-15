from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger

REMOVE_POS = {'P', 'Caa', 'Cab', 'Cba', 'Cbb'}
BATCH_SIZE = 5000
NUM_WORKERS = 4

ws_driver = CkipWordSegmenter(model='bert-base', device=-1)
pos_driver = CkipPosTagger(model='bert-base', device=-1)


def tokenize(text: str) -> list:
    ws_result = ws_driver([text])
    pos_result = pos_driver(ws_result)

    tokens = [
        word
        for word, pos in zip(ws_result[0], pos_result[0])
        if pos not in REMOVE_POS and word.strip()
    ]

    return tokens
