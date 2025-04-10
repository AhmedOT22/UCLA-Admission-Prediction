from src.config import EPOCHS, BATCH_SIZE
import os

def train_model(model, X_train, y_train):
    """train_model - trains the given model using the provided training data.

    Args:
        model (_type_): _description_
        X_train (_type_): _description_
        y_train (_type_): _description_

    Returns:
        _type_: _description_
    """
    model.fit(X_train, y_train)
    return model
