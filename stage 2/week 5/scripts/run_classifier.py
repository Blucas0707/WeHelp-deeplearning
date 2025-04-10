import argparse
import os
import numpy as np

from gensim.models.doc2vec import Doc2Vec
from sklearn.model_selection import train_test_split

from data_utils.loader import load_data
from models.classifier_trainer import (
    encode_labels,
    train_classifier,
    evaluate_classifier,
)
from data_utils.vector_infer import tokens_to_vectors_mp
from data_utils.vector_cache import get_vector_cache_path


def run_with_model(
    model_path: str,
    texts: list,
    labels: list,
    force_infer: bool = False,
    hidden_dims: list = [128, 64],
    epochs: int = 200,
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

    model = train_classifier(
        X_train,
        y_train,
        input_dim=d2v_model.vector_size,
        output_dim=len(encoder.classes_),
        hidden_dims=hidden_dims,
        epochs=epochs,
    )
    evaluate_classifier(model, X_test, y_test, encoder)


def main():
    parser = argparse.ArgumentParser(description='Doc2Vec + Classifier Pipeline')
    parser.add_argument(
        '--csv',
        type=str,
        default='tokenized_data.csv',
        help='Path to tokenized labeled CSV file',
    )
    parser.add_argument(
        '--model', type=str, required=True, help='Path to .model file or directory'
    )
    parser.add_argument(
        '--train_doc2vec',
        action='store_true',
        help='Train new Doc2Vec model if not loading',
    )
    parser.add_argument(
        '--force_infer',
        action='store_true',
        help='Force re-infer vectors even if cached',
    )
    parser.add_argument(
        '--epochs', type=int, default=200, help='Training epochs for classifier'
    )
    parser.add_argument(
        '--hidden_dims',
        type=str,
        default='128,64',
        help='Comma-separated hidden layer sizes',
    )

    args = parser.parse_args()
    texts, labels = load_data(args.csv)
    hidden_dims = list(map(int, args.hidden_dims.split(',')))

    if os.path.isdir(args.model):
        print(f'📁 Directory detected: {args.model}, running on all .model files')
        for fname in os.listdir(args.model):
            if fname.endswith('.model'):
                run_with_model(
                    os.path.join(args.model, fname),
                    texts,
                    labels,
                    hidden_dims=hidden_dims,
                    force_infer=args.force_infer,
                    epochs=args.epochs,
                )
    else:
        run_with_model(
            args.model,
            texts,
            labels,
            force_infer=args.force_infer,
            hidden_dims=hidden_dims,
            epochs=args.epochs,
        )


if __name__ == '__main__':
    main()
