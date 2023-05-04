from ._ezsave import ezsave
from ._ezload import ezload

# from basicpy import BaSiC


def save_illumination_correction_model(correction_model, file: str) -> None:
    ezsave(
        {"model": correction_model, "dummy": []},
        file,
    )


def load_illumination_correction_model(file: str):
    return ezload(file)["model"]
