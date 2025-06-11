DEFAULT_CONFIG = {
    "n_hidden": 15,
    "n_heads": 5,
    "max_epoch": 500,
    "lr": 0.01,
    "n_sample": None,
    "batch_col": None,
    "preprocess": False,
    "distance_set": {"start": 15, "stop": 60, "step": 5},
    "knn_set": {"start": 5, "stop": 40, "step": 5},
    "lambda_set": [
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
    ],
    "resolution_set": {"start": 0.01, "stop": 1.2, "step": 0.02},
    "n_repeats": 5,
    "spatial_graph_method": "distance",
    "repeated_negative_sampling": False,
    "y_aggregated": False,
}


class Config(dict):
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
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
        super().__setitem__(key, value)

    def to_dict(self):
        return {k: v.to_dict() if isinstance(v, Config) else v for k, v in self.items()}


def load_config(nested: dict = DEFAULT_CONFIG) -> Config:
    """Convert a (possibly nested) plain dict into a Config."""
    return Config(nested)
