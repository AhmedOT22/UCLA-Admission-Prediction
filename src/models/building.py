from sklearn.neural_network import MLPClassifier

from src.config import INPUT_DIM

def build_ann():
    model = MLPClassifier(
        hidden_layer_sizes=(3, 3),
        batch_size=50,
        max_iter=200,
        random_state=123
    )
    return model
