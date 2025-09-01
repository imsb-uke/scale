from dataclasses import dataclass, field
from typing import Literal

class BaseConfig(dict):
    """Dict that also supports attribute access (recursively)."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.update(*args, **kwargs)

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v

    # allow attr access
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(*e.args)

    # allow attr assignment
    def __setattr__(self, key, value):
        self[key] = value

    # ensure nested dicts become Config too
    def __setitem__(self, key, value):
        if isinstance(value, dict) and not isinstance(value, BaseConfig):
            value = BaseConfig(value)
        super().__setitem__(key, value)

    def to_dict(self):
        # Support both dataclass fields (if present) and dynamic keys stored in the dict
        result = {}
        # If this instance is a dataclass, prefer iterating its declared fields
        dataclass_fields = getattr(self, "__dataclass_fields__", None)
        if dataclass_fields:
            for name in dataclass_fields.keys():
                value = (
                    object.__getattribute__(self, name)
                    if not dict.__contains__(self, name)
                    else dict.__getitem__(self, name)
                )
                result[name] = (
                    value.to_dict() if isinstance(value, BaseConfig) else value
                )
        # Also include any extra dynamic keys that may have been added after init
        for k, v in dict.items(self):
            if k in result:
                continue
            result[k] = v.to_dict() if isinstance(v, BaseConfig) else v
        return result

    def __getattribute__(self, key):
        # if the key is stored in the dict, prefer it
        if key != "__dict__" and dict.__contains__(self, key):
            return dict.__getitem__(self, key)
        # otherwise fall back to the usual lookup chain
        return object.__getattribute__(self, key)


@dataclass
class Config(BaseConfig):
    seed: int = 200
    n_hidden: int = 15
    n_heads: int = 5
    max_epoch: int = 500
    lr: float = 0.01
    n_sample: int = None # number of maximum edges in case of distance graph (randomyl selected)
    sample_key: str = None
    preprocess: bool = False
    device: str | None = None
    distance_set: dict | list = field(
        default_factory=lambda: {"start": 15, "stop": 60, "step": 5}
    )
    knn_set: dict | list = field(default_factory=lambda: {"start": 5, "stop": 40, "step": 5})
    lambda_set: list = field(
        default_factory=lambda: [
            1e-6,
            5e-6,
            1e-5,
            5e-5,
            1e-4,
            5e-4,
            1e-3,
            5e-3,
            1e-2,
            5e-2,
            0.1,
            0.5,
            1,
            5,
            10,
        ]
    )
    resolution_set: dict = field(
        default_factory=lambda: {"start": 0.01, "stop": 1.2, "step": 0.02}
    )
    n_repeats: int = 5
    spatial_graph_method: Literal["distance", "knn"] = "distance"
    repeated_negative_sampling: bool = False
    y_aggregated: bool = False


def load_config(**kwargs):
    return Config(**kwargs)
