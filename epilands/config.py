from sre_parse import SPECIAL_CHARS
import os

NAME_SEPARATOR_ = "_"
COLUMN_SEPARATOR = "_"
DEFAULT_IMAGE_FORMATS = ["pdf"]
DEFAULT_DPI = 400


_fov_consecutive_to_rowcol_dict = {
    "1": "11",
    "2": "12",
    "3": "13",
    "4": "21",
    "5": "22",
    "6": "23",
    "7": "31",
    "8": "32",
    "9": "33",
}

_current_dir = os.path.dirname(os.path.abspath(__file__))

ALL_SUBDIRS = [
    d for d in os.listdir(_current_dir) 
    if (os.path.isdir(os.path.join(_current_dir, d)) and not d.startswith("__"))
    ]