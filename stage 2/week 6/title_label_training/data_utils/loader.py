import csv
from typing import List, Tuple
from gensim.models.doc2vec import TaggedDocument


def load_tokenized_documents(csv_file_path: str) -> List[TaggedDocument]:
    documents = []
    with open(csv_file_path, 'r', encoding='utf-8') as f:
        for idx, row in enumerate(csv.reader(f)):
            if len(row) < 2:
                continue
            documents.append(TaggedDocument(words=row[1:], tags=[str(idx)]))
    return documents


def load_data(csv_path: str) -> Tuple[List[List[str]], List[str]]:
    texts, labels = [], []
    with open(csv_path, 'r', encoding='utf-8') as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            labels.append(row[0])
            texts.append(row[1:])
    return texts, labels
