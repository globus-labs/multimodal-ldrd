"""Code related to doing the NMF in Keras"""
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Model
from keras import backend as K
from keras.layers import Layer
from keras.constraints import NonNeg


class SingleModalFactorization(Layer):
    """A non-negative matrix factorization that allows the constituent signals
    to shift across different measurements of the signal"""

    def __init__(self, n_components=2, fit_shift=False, **kwargs):
        """
        Args:
             n_components (int): Number of components to learn
             fit_shift (bool): Whether to fit a shift on the
        """
        super().__init__(**kwargs)
        self.n_components = n_components
        self.fit_shift = fit_shift

        # Placeholders for the weights
        self.components = self.contributions = self.shift = None

        # Placeholders for input shape information
        self.n_samples = self.n_channels = None

    def build(self, input_shape):
        # Process the inputs
        assert not any(i is None for i in input_shape), \
            "Input shape must be completely specified"
        self.n_samples, self.n_channels = input_shape

        # Add the training weights
        self.components = self.add_weight(
            name='components',
            shape=(self.n_components, self.n_channels),
            constraint=NonNeg(),
            initializer='uniform'
        )
        self.contributions = self.add_weight(
            name='contributions',
            shape=(self.n_samples, self.n_components),
            constraint=NonNeg(),
            initializer='zeros'
        )
        if self.fit_shift:
            self.shift = self.add_weight(
                name='shift',
                shape=(self.n_samples, self.n_components),
                initializer='zeros'
            )

    def call(self, inputs, **kwargs):
        if self.fit_shift:
            # Compute the shift matrix
            #  The shift matrix is a matrix where
            #   shift[i, j] = j - i
            #  Allows you to easily compute the distance of a specific
            #  point from all points in a pattern. We can use that
            #  distance to create interpolators
            # TODO (wardlt): Replace this with a spline interpolator?
            k_matrix = np.zeros((self.n_channels, self.n_channels))
            for k in range(-self.n_channels, self.n_channels, 1):
                kiu = np.triu_indices_from(k_matrix, k)
                k_matrix[kiu] = k
            k_matrix = k_matrix[None, None, :, :]
            k_matrix = np.tile(k_matrix, (self.n_samples, self.n_components, 1, 1))
            k_matrix = K.variable(k_matrix)

            # Compute the distance of each shift point from
            #  every point in each NMF component
            tiled_shift_matrix = K.expand_dims(K.expand_dims(self.shift, -1), -1)
            tiled_shift_matrix = K.tile(tiled_shift_matrix,
                                        (1, 1, self.n_channels, self.n_channels))
            distance_matrix = K.abs(tiled_shift_matrix - k_matrix)

            # Use the distances to compute interpolation coefficients
            #  For now, we use interpolation
            #   w_i = 1 - distance if distance < 1 else 0
            coeff_matrix = K.clip(1 - distance_matrix, 0, 1)

            # Multiply the coefficient matrix by the components to
            #  compute the interpolated components
            #   components = NC x NCh
            #   coeff_matrix = NS x NC x NCh X NCh
            #  where NC is number of components, NS is the number ofsamples,
            #  and NCh is the number of channels in the signal
            padded_components = K.expand_dims(K.expand_dims(self.components, -1), 0)
            self.shifted_components = K.squeeze(tf.matmul(
                coeff_matrix,
                padded_components
            ), 3)

            # Multiply the shifted coefficients by the contribution per component
            #  The shifted component matrix is NS x NC x NCh
            #  The contributions matrix is NS x NC
            self.shifted_contributions = tf.matmul(
                K.expand_dims(self.contributions, 1),
                self.shifted_components
            )

            # Now, sum up the contributions
            #  The shifted contribution matrix is NS x NC x NCh
            return K.sum(self.shifted_contributions, axis=1, keepdims=False)
        else:
            # Matrix multiply the contributes with the components
            #  Contributions is: NS x NC
            #  Components is: NC x NCh
            # Result: NS x NCh
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
