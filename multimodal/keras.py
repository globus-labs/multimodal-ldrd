"""Code related to doing the NMF in Keras"""
import numpy as np
import pandas as pd
from keras.models import Model
from keras import backend as K
from keras.layers import Layer
from keras.constraints import NonNeg
from keras.callbacks import EarlyStopping


class SingleModalFactorization(Layer):

    def __init__(self, n_components=2, **kwargs):
        super().__init__(**kwargs)
        self.n_components = n_components

        # Placeholders for the weights
        self.components = self.contributions = None

    def build(self, input_shape):
        # Process the inputs
        assert not any(i is None for i in input_shape), \
            "Input shape must be completely specified"
        n_samples, n_channels = input_shape

        # Add the training weights
        self.components = self.add_weight(
            name='components',
            shape=(self.n_components, n_channels),
            constraint=NonNeg(),
            initializer='uniform'
        )
        self.contributions = self.add_weight(
            name='contributions',
            shape=(n_samples, self.n_components),
            constraint=NonNeg(),
            initializer='zeros'
        )

    def call(self, inputs, **kwargs):
        return K.dot(self.contributions, self.components)

    def get_contributions(self) -> np.ndarray:
        return K.get_value(self.contributions)

    def get_components(self) -> np.ndarray:
        return K.get_value(self.components)


def run_training(factorizer: Model, signal: np.ndarray, max_epochs: int,
                 steps_per_epoch: int = 8,
                 early_stopping_patience: int = 20,
                 early_stopping_tolerance: float = 1e-6) -> pd.DataFrame:
    """Train the factorization model

    Args:
        factorizer (Model): Factorization tool
        signal (ndarray): Signal to be factorized
        max_epochs (int): Maximum number of epochs to run
        steps_per_epoch (int): Number of steps to for each epochs
        early_stopping_patience (int): Number of epochs loss has not improved
            by ``early_stopping_tolerance``
        early_stopping_tolerance (float): Absolute tolerance for early stopping.
            Loss must improved by more than this value before patience runs out
    Returns:
        (pd.DataFrame) Loss as a function of epoch
    """
    signal_tensor = K.variable(signal)  # Convert to Tensor for efficiency

    # Run the training
    losses = []
    loss = 0
    best_loss = np.inf
    since_best = 0
    for _ in range(max_epochs):
        # Run an epoch
        for _ in range(steps_per_epoch):
            loss = factorizer.train_on_batch(signal_tensor, signal_tensor)
        losses.append(loss)

        # Store the best loss
        since_best += 1
        if loss < best_loss - early_stopping_tolerance:
            best_loss = loss
            since_best = 0
        if since_best > early_stopping_patience:
            break
    return pd.DataFrame.from_dict({'epoch': list(range(len(losses))),
                                   'loss': losses})
