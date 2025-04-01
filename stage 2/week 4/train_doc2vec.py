import argparse
import os
from doc2vec_utils import (
    load_tokenized_documents,
    train_doc2vec_model,
    evaluate_model,
    grid_search_doc2vec,
)


def main():
    parser = argparse.ArgumentParser(description='Train or Grid Search Doc2Vec Model')
    parser.add_argument(
        '--csv', type=str, required=True, help='Path to tokenized CSV file'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default='models',
        help='Directory to save trained models',
    )
    parser.add_argument(
        '--save_model',
        action='store_true',
        help='Save trained model if similarity >= 80%',
    )
    parser.add_argument(
        '--grid_search', action='store_true', help='Enable grid search mode'
    )
    parser.add_argument(
        '--grid_size',
        type=str,
        choices=['small', 'full'],
        default='small',
        help='Size of parameter grid',
    )
    parser.add_argument(
        '--vector_size', type=int, default=100, help='Vector size for Doc2Vec'
    )
    parser.add_argument(
        '--epochs', type=int, default=100, help='Number of training epochs'
    )
    parser.add_argument('--min_count', type=int, default=2, help='Minimum token count')
    parser.add_argument('--window', type=int, default=5, help='Context window size')
    parser.add_argument(
        '--save_csv',
        type=str,
        default='grid_search_results.csv',
        help='Grid search result output',
    )

    args = parser.parse_args()

    print('📥 Loading documents...')
    documents = load_tokenized_documents(args.csv)

    if args.grid_search:
        print('🚀 Running Grid Search...')
        param_grid = {
            'vector_size': [50, 100, 150]
            if args.grid_size == 'small'
            else [50, 100, 150, 200],
            'epochs': [50, 100] if args.grid_size == 'small' else [50, 100, 150],
            'min_count': [1, 2],
            'window': [3, 5],
        }
        results_df = grid_search_doc2vec(documents, param_grid, save_dir=args.save_dir)
        results_df.to_csv(args.save_csv, index=False)
        print(f'✅ Grid search complete. Results saved to {args.save_csv}')
    else:
        print('🛠 Training Doc2Vec model...')
        model = train_doc2vec_model(
            documents,
            vector_size=args.vector_size,
            epochs=args.epochs,
            min_count=args.min_count,
            window=args.window,
        )

        print('📊 Evaluating model...')
        sim1, sim2 = evaluate_model(model, documents)
        print(f'Self Similarity@1: {sim1:.2f}%')
        print(f'Self Similarity@2: {sim2:.2f}%')

        if args.save_model and len(documents) >= 1000 and sim2 >= 80.0:
            model_name = f'd2v_v{args.vector_size}_e{args.epochs}_m{args.min_count}_w{args.window}.model'
            model_path = os.path.join(args.save_dir, model_name)
            os.makedirs(args.save_dir, exist_ok=True)
            model.save(model_path)
            print(f'💾 Model saved to {model_path}')
        else:
            print('🛑 Model not saved (condition not met or save_model not set)')


if __name__ == '__main__':
    main()
