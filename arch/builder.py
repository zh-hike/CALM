from .calm import CALM


def build(*args, **kwargs):
    model = CALM(*args, **kwargs)

    return model
