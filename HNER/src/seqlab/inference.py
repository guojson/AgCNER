from .train_model import ModelTrainer


def load_model(path):
    model = ModelTrainer.load_from_checkpoint(path).model
    model = model.eval()
    return model