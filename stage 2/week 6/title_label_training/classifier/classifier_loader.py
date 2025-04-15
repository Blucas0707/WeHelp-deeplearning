import torch
from title_label_training.classifier.schemas import TitleClassifier

def load_model(model_path):
    checkpoint = torch.load(model_path, map_location='cpu')
    model = TitleClassifier(
        input_dim=checkpoint['input_dim'],
        hidden_dims=checkpoint['hidden_dims'],
        output_dim=checkpoint['output_dim']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    encoder = checkpoint['label_encoder']
    return model, encoder
