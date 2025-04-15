from gensim.models import Doc2Vec
from title_label_training.classifier.classifier_loader import load_model
from title_label_training.tokenizer.tokenizer import tokenize
from title_label_training.data_utils.vector_infer import get_infer_vector_fn

D2V_MODEL_PATH = 'title_label_training/doc2vec_models/d2v_v50_e100_m2_w3.model'
MODEL_PATH = 'title_label_training/classifier/models/d2v_v50_e100_m2_w3.pt'


def predict_label(
    title: str,
    model_path: str = MODEL_PATH,
    d2v_model_path: str = D2V_MODEL_PATH,
    method='doc2vec',
) -> str:
    model, encoder = load_model(model_path)
    d2v_model = Doc2Vec.load(d2v_model_path)
    tokens = tokenize(title)
    infer_vector_fn = get_infer_vector_fn(method)
    vector = infer_vector_fn(d2v_model, tokens).reshape(1, -1)
    pred_idx = model.predict(vector)[0]
    return encoder.inverse_transform([pred_idx])[0]


def get_suggested_categories(predicted: str) -> list:
    # TODO: 根據預測結果動態給建議分類
    return [
        'Baseball',
        'Boy-Girl',
        'C_Chat',
        'HatePolitics',
        'Lifismoney',
        'Military',
        'PC_Shopping',
        'Stock',
        'Tech_Job',
    ]
