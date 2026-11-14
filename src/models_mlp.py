from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from .data import make_preprocessor

def make_mlp(hidden_layer_sizes=(16,), alpha=1e-4, learning_rate_init=1e-3, max_iter=300):
    """Build an untrained MLP pipeline: preprocessing followed by an MLPClassifier.

    The preprocessor (standard scaling + one-hot encoding of Type) lives inside
    the pipeline, so it is fitted only on the training rows of each fold.

    Early stopping is off (it monitors accuracy, which fails with imbalanced
    classes) and random_state is fixed; only the arguments below change.


    Parameters
    ----------
    hidden_layer_sizes : tuple of int, default=(16,)
        Neurons in each hidden layer, e.g. (32, 16) is two layers.
    alpha : float, default=1e-4
        L2 regularisation strength. Larger values give simpler models.
    learning_rate_init : float, default=1e-3
        Initial step size of the optimiser.
    max_iter : int, default=300
        Maximum number of epochs. Training may stop earlier if the training loss stops improving.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Untrained pipeline, ready to be passed to evaluate().

    Example
    -------
    >>> model = make_mlp(hidden_layer_sizes=(32, 16), alpha=1e-3, max_iter=500)
    """

    # Create the MLP classifier with the params
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes, # Size of the hidden layers
        alpha=alpha,    # to avoid overfitting
        learning_rate_init=learning_rate_init,
        early_stopping=False, # off: its accuracy-based stop cut training too early (see baseline)
        max_iter=max_iter, # number max of epochs. 
        random_state=42 # fixed for reproducibility
    )
    return make_pipeline(make_preprocessor(), mlp)
