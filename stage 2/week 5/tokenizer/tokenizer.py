import pandas as pd
import concurrent.futures
from typing import List, Iterator
from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger

REMOVE_POS = {'P', 'Caa', 'Cab', 'Cba', 'Cbb'}
BATCH_SIZE = 5000
NUM_WORKERS = 4

ws_driver = CkipWordSegmenter(model='bert-base', device=-1)
pos_driver = CkipPosTagger(model='bert-base', device=-1)


def read_csv_in_chunks(
    input_file: str, chunksize: int = BATCH_SIZE
) -> Iterator[pd.DataFrame]:
    return pd.read_csv(input_file, chunksize=chunksize)


def process_batch(sentences: List[str]) -> List[List[str]]:
    ws_results = ws_driver(sentences)
    pos_results = pos_driver(ws_results)

    return [
        [
            word
            for word, pos in zip(words, pos_tags)
            if pos not in REMOVE_POS and word.strip()
        ]
        for words, pos_tags in zip(ws_results, pos_results)
    ]


def process_csv_stream(input_file: str, output_file: str) -> None:
    with open(output_file, 'w', encoding='utf-8') as f:
        for chunk in read_csv_in_chunks(input_file):
            labels = chunk['board_name'].tolist()
            sentences = chunk['title'].tolist()

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=NUM_WORKERS
            ) as executor:
                processed_sentences = list(executor.map(process_batch, [sentences]))

            for label, words in zip(labels, processed_sentences[0]):
                f.write(','.join([label] + words) + '\n')

    print(f'處理完成，結果已存至 {output_file}')


def main(input_file: str, output_file: str) -> None:
    process_csv_stream(input_file, output_file)


if __name__ == '__main__':
    input_file = 'sample_cleaned_data.csv'
    output_file = 'tokenized_data.csv'
    main(input_file, output_file)
