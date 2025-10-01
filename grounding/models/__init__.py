from .gpt_oss import GPTOSS
from .model import Model


__all__ = ["get_model"]

__model_map__ = {
    "gpt-oss": GPTOSS,
}

def get_model(model_name: str, *args, **kwargs) -> Model:
    if model_name in __model_map__:
        return __model_map__[model_name](*args, **kwargs)
    else:
        raise ValueError(f"Model {model_name} not found")
