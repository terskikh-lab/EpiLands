import numpy as np
from typing import Tuple
from ._get_object_bbox import get_object_bbox


# Description cellselecter
# separate cells from the image
# Kenta Ninomiya @ Kyushu University: 2021/07/29
def cellselecter(
    img: np.ndarray, label: np.ndarray, margin: int, cellIdx: int
) -> tuple:
    """
    Selects a single cell from an image based on its label.

    Args:
    - img (np.ndarray): The input image.
    - label (np.ndarray): The label matrix of the image.
    - margin (int): The number of pixels to pad around the selected cell.
    - cellIdx (int): The index of the cell to select.

    Returns:
    - tuple: A tuple containing the selected cell image and its label matrix.
    """

    # get the binary image of the "celIdxl"-th cell
    objCellLabel = np.where(
        label == cellIdx, 1, 0
    )  # set teh value one to the "celIdxl"-th cell, zero for the others

    # get the size of the image
    [rowNum, colNum] = objCellLabel.shape

    # get a maximum and minimum row coordinate
    coordinateRow = np.arange(0, rowNum)
    idxRow = np.any(a=objCellLabel == 1, axis=int(1))

    # get a maximum and minimum column coordinate
    coordinateCol = np.arange(0, colNum)
    idxCol = np.any(a=objCellLabel == 1, axis=int(0))

    rowMin = coordinateRow[idxRow][0]
    rowMax = coordinateRow[idxRow][-1]
    colMin = coordinateCol[idxCol][0]
    colMax = coordinateCol[idxCol][-1]

    # slicing the matrix
    objImg = img[rowMin : rowMax + 1, colMin : colMax + 1]
    objImg = np.pad(objImg, [margin, margin], "constant")

    objCellLabel = objCellLabel[rowMin : rowMax + 1, colMin : colMax + 1]
    objCellLabel = np.pad(objCellLabel, [margin, margin], "constant")

    return objImg, objCellLabel


def cellselecter_ND(
    img: np.ndarray,
    label: np.ndarray,
    margin: int,
    cellIdx: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Selects a specific cell from an image and its corresponding label.

    Args:
        img (np.ndarray): The input image.
        label (np.ndarray): The label of the input image.
        margin (int): The margin to add around the selected cell.
        cellIdx (int): The index of the cell to select.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the selected cell image and its corresponding label.
    """
    # get the binary image of the "celIdxl"-th cell
    objCellLabel = np.where(
        label == cellIdx, 1, 0
    )  # set teh value one to the "celIdxl"-th cell, zero for the others
    objectBBox = get_object_bbox(objCellLabel)
    # slicing the matrix
    slices = tuple(slice(i, j) for (i, j) in objectBBox)
    objImg = img[slices]
    objImg = np.pad(objImg, [margin, margin], "constant")
    objCellLabel = objCellLabel[slices]
    objCellLabel = np.pad(objCellLabel, [margin, margin], "constant")
    return objImg, objCellLabel
