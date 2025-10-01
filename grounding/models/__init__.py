from .model import Model

__all__ = ["get_model", "Model"]


def get_model(model_name: str, *args, **kwargs) -> Model:
    from .gpt_oss import GPTOSS

    __model_map__ = {
        "gpt-oss": GPTOSS,
    }

    if model_name in __model_map__:
        return __model_map__[model_name](*args, **kwargs)
    else:
        raise ValueError(f"Model {model_name} not found")
