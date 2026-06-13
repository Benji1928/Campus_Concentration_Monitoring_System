"""
MLP configuration classes for training experiments.

Three variants, each saved to a separate output directory:
  MLPBaselineConfig  →  models/MLP/         (matches train_evaluate.py exactly)
  MLPV1Config        →  models/MLP_V1/
  MLPV2Config        →  models/MLP_V2/

Used by train_evaluate_v1.py and train_evaluate_v2.py.
"""


class MLPBaselineConfig:
    """Current parameters — kept identical to train_evaluate.py."""
    description = "Baseline: 2-layer [64,32], relu, Adam, no regularization"
    output_dir = "MLP"

    hidden_layer_sizes = (64, 32)
    activation = "relu"
    solver = "adam"
    alpha = 0.0001
    learning_rate_init = 0.001
    max_iter = 500
    random_state = 42
    early_stopping = False


class MLPV1Config:
    """
    V1: Deeper network, relu, LR=1e-3, weight_decay=1e-4.
    Shared with CNN_training_v1.py: same activation, learning_rate, weight_decay (alpha).
    """
    description = "V1: 3-layer [128,64,32], relu, Adam, LR=1e-3, alpha=1e-4, early_stopping"
    output_dir = "MLP_V1"

    hidden_layer_sizes = (128, 64, 32)
    activation = "relu"
    solver = "adam"
    alpha = 1e-4             # matches CNN V1 weight_decay
    learning_rate_init = 1e-3  # matches CNN V1 learning_rate
    max_iter = 1000
    random_state = 42
    early_stopping = True
    validation_fraction = 0.1
    n_iter_no_change = 15


class MLPV2Config:
    """
    V2: Wider network, tanh, LR=5e-4, weight_decay=5e-4.
    Shared with CNN_training_v2.py: same activation, learning_rate, weight_decay (alpha).
    """
    description = "V2: 3-layer [256,128,64], tanh, Adam, LR=5e-4, alpha=5e-4, early_stopping"
    output_dir = "MLP_V2"

    hidden_layer_sizes = (256, 128, 64)
    activation = "tanh"
    solver = "adam"
    alpha = 5e-4             # matches CNN V2 weight_decay
    learning_rate_init = 5e-4  # matches CNN V2 learning_rate
    max_iter = 500
    random_state = 42
    early_stopping = True
    validation_fraction = 0.1
    n_iter_no_change = 20
