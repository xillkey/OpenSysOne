"""OpenSysOne: train System-1 decision models (Laya / Jev style) on any encoder backbone."""

from .backbone import BACKBONES, Backbone, HFEncoder, build_backbone, register_backbone
from .schema import Example, Question, SchemaError, load_jsonl, parse_example
from .serializer import OPT_TOKEN, Q_TOKEN, Serializer, SerializerConfig

__version__ = "0.1.0"

__all__ = [
    "BACKBONES", "Backbone", "HFEncoder", "build_backbone", "register_backbone",
    "Example", "Question", "SchemaError", "load_jsonl", "parse_example",
    "Serializer", "SerializerConfig", "Q_TOKEN", "OPT_TOKEN",
    "DecisionModel", "Predictor", "train", "load_config", "__version__",
]


def __getattr__(name: str):  # lazy imports: avoid loading training/serving code until used
    if name == "DecisionModel":
        from .model import DecisionModel
        return DecisionModel
    if name == "Predictor":
        from .predictor import Predictor
        return Predictor
    if name == "train":
        from .trainer import train
        return train
    if name == "load_config":
        from .config import load_config
        return load_config
    raise AttributeError(name)
