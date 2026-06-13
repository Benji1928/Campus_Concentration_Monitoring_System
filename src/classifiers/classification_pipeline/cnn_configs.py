"""
CNN configuration classes for training experiments.

Three variants, each saved to a separate output directory:
  CNNBaselineConfig  →  CNN_training.ipynb (unchanged — MobileNetV3, LR=1e-4, BS=32)
  CNNV1Config        →  CNN_training_v1.py (EfficientNet-B0, LR=3e-4, CosineAnnealingLR)
  CNNV2Config        →  CNN_training_v2.py (MobileNetV3, heavier regularization, label smoothing)

Set dataset_dir to your local dataset path when not running on Kaggle.
"""


class CNNBaselineConfig:
    """Matches CNN_training.ipynb exactly — kept untouched as the reference."""
    description = "Baseline: MobileNetV3, LR=1e-4, BS=32, patience=3, head=[128], dropout=0.3"
    output_dir = "CNN_Baseline"

    model_name = "mobilenetv3_large_100"
    batch_size = 32
    epochs = 15
    learning_rate = 1e-4
    weight_decay = 0.0
    dropout = 0.3
    head_hidden = [128]
    patience = 3
    min_delta = 0.001
    test_size = 0.15
    num_classes = 3
    label_smoothing = 0.0
    use_cosine_lr = False
    extra_augmentation = False

    # Dataset path — Kaggle default; override for local runs
    dataset_dir = "/kaggle/input/datasets/natalierobert/designproject-clean"


class CNNV1Config:
    """
    V1: EfficientNet-B0 backbone with deeper head and CosineAnnealingLR.
    Shared with MLP_V1: same activation (relu), learning_rate (1e-3), weight_decay (1e-4).
    """
    description = (
        "V1: EfficientNet-B0, relu, LR=1e-3, BS=64, patience=5, "
        "head=[256,128], dropout=0.4, CosineAnnealingLR, weight_decay=1e-4"
    )
    output_dir = "CNN_V1"

    model_name = "efficientnet_b0"
    batch_size = 64
    epochs = 30
    activation = "relu"           # matches MLPV1Config.activation
    learning_rate = 1e-3          # matches MLPV1Config.learning_rate_init
    weight_decay = 1e-4           # matches MLPV1Config.alpha
    dropout = 0.4
    head_hidden = [256, 128]
    patience = 5
    min_delta = 0.001
    test_size = 0.2
    num_classes = 3
    label_smoothing = 0.0
    use_cosine_lr = True
    extra_augmentation = False

    dataset_dir = "/kaggle/input/datasets/natalierobert/designproject-clean"


class CNNV2Config:
    """
    V2: MobileNetV3 with tanh head, heavy regularization and label smoothing.
    Shared with MLP_V2: same activation (tanh), learning_rate (5e-4), weight_decay (5e-4).
    """
    description = (
        "V2: MobileNetV3, tanh, LR=5e-4, BS=32, patience=7, "
        "head=[512,256], dropout=0.5, label_smooth=0.1, weight_decay=5e-4"
    )
    output_dir = "CNN_V2"

    model_name = "mobilenetv3_large_100"
    batch_size = 32
    epochs = 30
    activation = "tanh"           # matches MLPV2Config.activation
    learning_rate = 5e-4          # matches MLPV2Config.learning_rate_init
    weight_decay = 5e-4           # matches MLPV2Config.alpha
    dropout = 0.5
    head_hidden = [512, 256]
    patience = 7
    min_delta = 0.0005
    test_size = 0.2
    num_classes = 3
    label_smoothing = 0.1
    use_cosine_lr = False
    extra_augmentation = True

    dataset_dir = "/kaggle/input/datasets/natalierobert/designproject-clean"
