from dataclasses import dataclass, field


@dataclass
class DeiTTinyConfig:
    model_name: str = 'deit_tiny_patch16_224'
    img_size: int = 224
    num_classes: int = 3
    head_hidden_sizes: list = field(default_factory=lambda: [128])
    dropout: float = 0.2
    activation: str = 'relu'
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 32
    epochs: int = 15
    patience: int = 5
    min_delta: float = 0.001
    use_cosine_lr: bool = True
    label_smoothing: float = 0.0
    test_size: float = 0.2
    dataset_dir: str = 'data/face_dataset'
    train_dir: str = ''
    val_dir: str = ''
    use_class_weights: bool = False
    class_counts: list = field(default_factory=lambda: [1179, 1025, 790])
    output_dir: str = 'DeiT_Tiny'


@dataclass
class MobileViTXXSConfig:
    model_name: str = 'mobilevit_xxs'
    img_size: int = 256
    num_classes: int = 3
    head_hidden_sizes: list = field(default_factory=lambda: [])
    dropout: float = 0.1
    activation: str = 'relu'
    lr: float = 2e-4
    weight_decay: float = 1e-4
    batch_size: int = 32
    epochs: int = 15
    patience: int = 5
    min_delta: float = 0.001
    use_cosine_lr: bool = True
    label_smoothing: float = 0.0
    test_size: float = 0.2
    dataset_dir: str = 'data/face_dataset'
    train_dir: str = ''
    val_dir: str = ''
    use_class_weights: bool = False
    class_counts: list = field(default_factory=lambda: [1179, 1025, 790])
    output_dir: str = 'MobileViT_XXS'
