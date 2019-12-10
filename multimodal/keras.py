"""Code related to doing the NMF in Keras"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from keras.models import Model
from keras import backend as K
from keras.layers import Layer
from keras.constraints import NonNeg
from keras.regularizers import Regularizer
from typing import List, Optional, Tuple, Union


def generate_signals(n_samples: int, n_components: int, n_channels: int,
                     contributions: tf.Tensor, components: tf.Tensor, shift: Optional[tf.Tensor],
                     subbatch_size: int = 32) -> Tuple[tf.Tensor, tf.Tensor]:
    """Apply the operations to generate the output signal given the contributions, components and shifts

    Args:
        n_samples (int): Number of samples in the experiment being analyzed
        n_channels (int): Number of channels in each signal
        n_components (int): Number of components to learn from the data
        contributions (tf.Tensor): Contribution level for each of the learned components
        components (tf.Tensor): Values of each of the learned components
        shift (tf.Tensor): Optional list of shift values
        subbatch_size (int): Number of samples for which to compute shapes at once
    Returns:
        total_signal (tf.Tensor): Signal generated for each sample. Shape: n_samples, n_channels
        per_sample_contributions (tf.Tensor): Component from each subshape. Shape: n_components, n_samples, n_channels
    """
    if shift is not None:
        # Make a shift matrix
        #  The shift matrix is a matrix where
        #   shift[i, j] = j - i
        #  Allows you to easily compute the distance of a specific
        #  point from all points in a pattern. We can use that
        #  distance to create interpolators
        k_matrix = np.zeros((n_channels, n_channels))
        for k in range(-n_channels, n_channels, 1):
            kiu = np.triu_indices_from(k_matrix, k)
            k_matrix[kiu] = k
        k_matrix = K.constant(k_matrix[None, :, :]) # Shape: 1, NCh, NCh

        # Compute the shifts from each component
        shifted_components = []
        for c in range(n_components):
            # Get the component and the contribution
            my_component = components[c, :]  # shape: (n_channels)
            my_contribution = contributions[:, c]  # shape: (n_samples)

            # Get the component in a ready to multiply form
            tiled_component = K.reshape(my_component, (1, n_channels, 1))  # Shape: 1 x NCh x 1

            # Shift each individual components
            #  Note: We process samples in chunks because there are memory
            #   limitations in TF that prevent running all samples concurrently
            #   (protobuf's limit of 2GB message sizes) and processing samples
            #   one-at-a-time is too slow (not much intra-operation parallelism)
            #  The self.subbatch_size is a tuning parameter
            my_shifted_components = []
            sample_chunks = np.array_split(list(range(n_samples)),
                                           n_samples // subbatch_size)
            for s_chunk in sample_chunks:
                # Get the shift values, tile to make ready for subtraction
                my_shift = tf.slice(shift,  # shape: (len(s_chunk), 1)
                                    begin=(min(s_chunk), c),
                                    size=(len(s_chunk), 1),
                                    name=f'component_{c}-samples_'
                                         f'{min(s_chunk)}-{max(s_chunk)}')
                my_shift = K.expand_dims(my_shift)  # shape: NS' x 1 x 1
                my_shift = K.tile(my_shift, (1, 1, n_channels))

                # Compute the distance from each point
                distance_matrix = K.abs(tf.subtract(k_matrix, my_shift))  # Shape: NS' x NCh x NCh
                coeff_matrix = K.clip(1 - distance_matrix, 0, 1)

                # Compute the shift
                shifted_component = tf.matmul(coeff_matrix, tiled_component)
                my_shifted_components.append(K.squeeze(shifted_component, -1))

            # Stack them to form: (n_samples, n_channels)
            my_shifted_components = tf.concat(my_shifted_components, axis=0,
                                              name=f'shifted_components_{c}')

            # Multiply by the contribution ot each sample
            my_shifted_contribution = tf.multiply(
                my_shifted_components,
                K.expand_dims(my_contribution, -1)
            )
            shifted_components.append(tf.stack(my_shifted_contribution))

        # Stacked array has a shape: (n_components, n_samples, n_channels)
        shifted_contributions = tf.stack(shifted_components, 0,
                                         name='shifted_contributions')

        # The total signal is the sum of all components
        total_signal = K.sum(shifted_contributions, axis=0, keepdims=False)
        return total_signal, shifted_contributions
    else:
        # Matrix multiply the contributes with the components
        #  Contributions is: NS x NC
        #  Components is: NC x NCh
        # Result: NS x NCh
        total_signal = K.dot(contributions, components)

        # Compute the contribution of each signal
        #  First expand the matrices so that broadcasting would work
        #  Contributions: NS x NC -T> NC x NS -ex_dim-> NC x NS x 1
        #  Components: NC x NCh -ex_dim-> NC x 1 x NCh
        # Output shape: NC x NS x NCh
        per_comp_contributions = tf.multiply(
            tf.expand_dims(tf.transpose(contributions), -1),
            tf.expand_dims(components, 1)
        )

        return total_signal, per_comp_contributions


class MultiRegularizer(Regularizer):
    """Employ multiple regularizers"""

    def __init__(self, regularizers: List[Regularizer]):
        """
        Args:
             regularizers ([Regularizers]): Regularizers to sum together
        """
        super().__init__()
        self.regularizers = regularizers

    def __call__(self, x):
        return sum(r(x) for r in self.regularizers)


class ContinuityRegularizer(Regularizer):
    """Applies a penalty to the change in weights along a certain axis

    The regularization strength is equal to the sum of (x[i] - x[i-1])^2.
    """

    def __init__(self, weight: float, axis: int = -1):
        """
        Args:
             weight (float): Strength of the regularization
             axis (int): Axis along which to compute differences
        """
        super().__init__()
        self.weight = weight
        self.axis = axis

    def __call__(self, x):
        # Identify the desired diff axis
        n_dim = K.ndim(x)
        diff_axis = self.axis if self.axis != -1 else n_dim - 1

        # Make two slices: one of [0, N-1] and another to [1, N] along the diff axis
        slice_0 = [slice(None) for _ in range(n_dim)]
        slice_0[diff_axis] = slice(0, -1)
        slice_1 = [slice(None) for _ in range(n_dim)]
        slice_1[diff_axis] = slice(1, None)

        # Get the values along the slice axis
        slice_0_values = x[slice_0]
        slice_1_values = x[slice_1]

        # Compute (x[i] - x[i-1]) * 2
        penalty = K.pow(slice_0_values - slice_1_values, 2)
        return self.weight * K.sum(penalty)


class SingleModalFactorization(Layer):
    """A non-negative matrix factorization that allows the constituent signals
    to shift across different measurements of the signal"""

    def __init__(self, n_components=2, fit_shift=False,
                 component_regularizer=None, subbatch_size=32,
                 contribution_regularizer=None,
                 shift_regularizer=None,
                 **kwargs):
        """
        Args:
             n_components (int): Number of components to learn
             fit_shift (bool): Whether to fit a shift on the
             component_regularizer (Regularizer): Regularizer for component signals
             contribution_regularizer (Regularizer): Regularizer for the
                contributions from component signals
             shift_regularizer (Regularizer): Regularizer for the shift factors
             subbatch_size (int): Number of samples to process concurrently
        """
        super().__init__(**kwargs)
        self.n_components = n_components
        self.fit_shift = fit_shift
        self.component_regularizer = component_regularizer
        self.contribution_regularizer = contribution_regularizer
        self.shift_regularizer = shift_regularizer
        self.subbatch_size = subbatch_size

        # Placeholders for the weights
        self.components = self.contributions = self.shift = None
        self.per_sample_contribution = None

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
            initializer='uniform',
            regularizer=self.component_regularizer
        )
        self.contributions = self.add_weight(
            name='contributions',
            shape=(self.n_samples, self.n_components),
            constraint=NonNeg(),
            initializer='uniform',
            regularizer=self.contribution_regularizer
        )
        if self.fit_shift:
            self.shift = self.add_weight(
                name='shift',
                shape=(self.n_samples, self.n_components),
                initializer='uniform',
                regularizer=self.shift_regularizer,
            )

    def call(self, inputs, **kwargs):
        total_signal, self.per_sample_contribution = generate_signals(
            self.n_samples, self.n_components, self.n_channels,
            self.contributions, self.components, self.shift,
            subbatch_size=self.subbatch_size
        )
        return total_signal

    def get_contributions(self) -> np.ndarray:
        """Get the contributions levels of each signal

        Returns:
            (ndarray) Contributions. Shape: n_samples, n_components
        """
        return K.get_value(self.contributions)

    def get_components(self) -> np.ndarray:
        """Get the learned components from the signal

        Returns:
            (ndarray) Components. Shape: n_components, n_channels
        """
        return K.get_value(self.components)

    def get_per_sample_contribution(self) -> np.ndarray:
        """Get the contribution to the signal for each component for each sample

        Returns:
            (ndarray): Contributions: Shape: n_components, n_samples, n_channels
        """
        return K.get_value(self.per_sample_contribution)


class MultiModalFactorizer(Layer):
    """Generates a multi-modal signal

    This layer stores different components and shifts for each mode but has a single contribution
    matrix for all modalities. The important point is that we learn component signals from all modalities
    that are related to each other.
    """

    def __init__(self, n_components=2, fit_shift=False,
                 component_regularizer=None, subbatch_size=32,
                 contribution_regularizer=None,
                 shift_regularizer=None,
                 **kwargs):
        """

        Args:
             n_components (int): Number of components to learn
             fit_shift ([bool]): Whether to fit a shift for each component
             component_regularizer (Regularizer): Regularizer for component signals
             contribution_regularizer (Regularizer): Regularizer for the
                contributions from component signals
             shift_regularizer (Regularizer): Regularizer for the shift factors
             subbatch_size (int): Number of samples to process concurrently
        """
        super().__init__(**kwargs)
        self.n_components = n_components
        self.fit_shift = fit_shift
        self.component_regularizer = component_regularizer
        self.contribution_regularizer = contribution_regularizer
        self.shift_regularizer = shift_regularizer
        self.subbatch_size = subbatch_size

        # Placeholders for the weights
        self.components = self.contributions = self.shift = None
        self.per_sample_contribution = None

        # Placeholders for input shape information
        self.n_modes = self.n_samples = self.n_channels = None

    def build(self, input_shape):
        self.n_modes = len(input_shape)

        # Create the learned components for each shape
        self.n_channels = []
        self.components = []
        self.shift = []
        for mode, (mode_shape, fit_shift) in enumerate(zip(input_shape, self.fit_shift)):
            # Get the shape of this layer
            assert not any(i is None for i in mode_shape), \
                "Input shape must be completely specified"
            self.n_samples, n_channels = mode_shape
            self.n_channels.append(n_channels)

            # Add the training weights
            components = self.add_weight(
                name=f'components_mode_{mode}',
                shape=(self.n_components, n_channels),
                constraint=NonNeg(),
                initializer='uniform',
                regularizer=self.component_regularizer
            )
            self.components.append(components)

            # Make shift matrix, if needed
            shift = None
            if fit_shift:
                shift = self.add_weight(
                    name=f'shift_mode_{mode}',
                    shape=(self.n_samples, self.n_components),
                    initializer='uniform',
                    regularizer=self.shift_regularizer,
                )
            self.shift.append(shift)

        # We will have a single contribution matrix for all modes
        self.contributions = self.add_weight(
            name='contributions',
            shape=(self.n_samples, self.n_components),
            constraint=NonNeg(),
            initializer='uniform',
            regularizer=self.contribution_regularizer
        )

    def call(self, inputs, **kwargs):
        self.per_sample_contribution = []
        outputs = []
        for n_channels, components, shift in zip(self.n_channels, self.components, self.shift):
            total_signal, per_sample_signal = generate_signals(
                self.n_samples, self.n_components, n_channels,
                self.contributions, components, shift,
                subbatch_size=self.subbatch_size
            )
            outputs.append(total_signal)
            self.per_sample_contribution.append(per_sample_signal)
        return outputs

    def get_contributions(self) -> np.ndarray:
        """Get the contributions levels of each component

        Returns:
            (ndarray) Contributions. Shape: n_samples, n_components
        """
        return K.get_value(self.contributions)

    def get_components(self, mode: int) -> np.ndarray:
        """Get the learned components from the signal

        Args:
            mode (int): Which mode
        Returns:
            (ndarray) Components. Shape: n_components, n_channels
        """
        return K.get_value(self.components[mode])

    def get_per_sample_contribution(self, mode: int) -> np.ndarray:
        """Get the contribution to the signal for each component for each sample

        Args:
            mode (int): Which mode
        Returns:
            (ndarray): Contributions: Shape: n_components, n_samples, n_channels
        """
        return K.get_value(self.per_sample_contribution[mode])


def run_training(factorizer: Model, signal: Union[np.ndarray, List[np.ndarray]],
                 max_epochs: int,
                 steps_per_epoch: int = 8,
                 early_stopping_patience: int = 20,
                 early_stopping_tolerance: float = 1e-6,
                 verbose=False) -> pd.DataFrame:
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
        verbose (bool): Whether to display a progress bar
    Returns:
        (pd.DataFrame) Loss as a function of epoch
    """
    # Convert to Tensor for efficiency
    if isinstance(signal, np.ndarray):
        signal_tensor = K.variable(signal)
        single_output = True
    else:
        signal_tensor = [K.variable(s) for s in signal]
        single_output = False

    # Run the training
    losses = []
    loss = 0
    best_loss = np.inf
    since_best = 0
    epoch_iter = tqdm(range(max_epochs), dynamic_ncols=True,
                      disable=not verbose)
    for _ in epoch_iter:
        # Run an epoch
        for _ in range(steps_per_epoch):
            loss = factorizer.train_on_batch(signal_tensor, signal_tensor)

        # Get only the entire loss
        if not single_output:
            loss = loss[0]
        losses.append(loss)

        # Store the best loss
        since_best += 1
        if loss < best_loss - early_stopping_tolerance:
            best_loss = loss
            since_best = 0
        if since_best > early_stopping_patience:
            break

        # Update the display
        epoch_iter.set_postfix({'best_loss': best_loss, 'loss': loss})
    return pd.DataFrame.from_dict({'epoch': list(range(len(losses))),
                                   'loss': losses})
