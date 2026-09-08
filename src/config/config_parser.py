import yaml
from dataclasses import dataclass, field, asdict
from typing import List, Tuple

@dataclass
class Config:
    # Model
    bottleneck_size: int = 4
    kernel_size: int = 4
    padding: int = 1
    fixed_layers: int = 5
    nz: int = 256
    ngf: int = 64
    ndf: int = 64
    image_size: int = 64
    nc: int = 1
    
    # Loss & Training
    lambda_l1: float = 1.0
    lambda_gp: float = 10.0
    lr: float = 0.0004
    batch_size: int = 32
    betas: Tuple[float, float] = (0.0, 0.9)
    num_epochs: int = 900
    n_critic: int = 2
    
    # Augmentation
    aug_scale: Tuple[float, float] = (0.7, 1.32)
    aug_translate: Tuple[float, float] = (0.32, 0.32)
    aug_mask_prob: float = 0.0
    aug_degrees: float = 0.0
    
    # Dataset
    missing_chars: List[str] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if "betas" in data and isinstance(data["betas"], list):
            data["betas"] = tuple(data["betas"])
        if "aug_scale" in data and isinstance(data["aug_scale"], list):
            data["aug_scale"] = tuple(data["aug_scale"])
        if "aug_translate" in data and isinstance(data["aug_translate"], list):
            data["aug_translate"] = tuple(data["aug_translate"])
        return cls(**data)
        
    def to_dict(self):
        return asdict(self)

