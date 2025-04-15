import argparse
import os
import numpy as np
import torch
from gensim.models.doc2vec import Doc2Vec
from sklearn.model_selection import train_test_split
from classifier.classifier_trainer import (
    encode_labels,
    train_classifier,
    evaluate_classifier,
)
from data_utils.loader import load_data
from data_utils.vector_infer import tokens_to_vectors_mp
from data_utils.vector_cache import get_vector_cache_path


def run_with_model(
    model_path, texts, labels, force_infer=False, hidden_dims=[128, 64], epochs=200
):
    print(f'\n📦 Loading Doc2Vec model: {model_path}')
    d2v_model = Doc2Vec.load(model_path)
    cache_path = get_vector_cache_path(model_path)

    if os.path.exists(cache_path) and not force_infer:
        print(f'✅ Cached vectors found: {cache_path}, loading...')
        vectors = np.load(cache_path)
    else:
        print('🚀 Inferring vectors...')
        vectors = tokens_to_vectors_mp(d2v_model, texts, workers=8)
        np.save(cache_path, vectors)
        print(f'💾 Saved vectors to: {cache_path}')

    y, encoder = encode_labels(labels)
    X_train, X_test, y_train, y_test = train_test_split(
        vectors, y, test_size=0.2, stratify=y, random_state=42
    )
    model, model_info = train_classifier(
        X_train,
        y_train,
        d2v_model.vector_size,
        len(encoder.classes_),
        hidden_dims,
        epochs,
    )
    evaluate_classifier(model, X_test, y_test, encoder)
    clf_path = model_path.replace('doc2vec_models', 'classifier/models').replace(
        '.model', '.pt'
    )
    os.makedirs(os.path.dirname(clf_path), exist_ok=True)
    torch.save({**model_info, 'label_encoder': encoder}, clf_path)
    print(f'💾 Saved classifier to: {clf_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='tokenized_data.csv')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--force_infer', action='store_true')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--hidden_dims', type=str, default='128,64')
    args = parser.parse_args()
    texts, labels = load_data(args.csv)
    hidden_dims = list(map(int, args.hidden_dims.split(',')))
    if os.path.isdir(args.model):
        for fname in os.listdir(args.model):
            if fname.endswith('.model'):
                run_with_model(
                    os.path.join(args.model, fname),
                    texts,
                    labels,
                    args.force_infer,
                    hidden_dims,
                    args.epochs,
                )
    else:
        run_with_model(
            args.model, texts, labels, args.force_infer, hidden_dims, args.epochs
        )


if __name__ == '__main__':
    main()
