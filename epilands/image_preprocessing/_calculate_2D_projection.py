import numpy as np
import tifffile as tiff


def calculate_2D_projection(files: list, method: str = "max") -> np.ndarray:
    imgStack = list()
    # imgStackShow=list()
    for file in files:
        tmpImg = tiff.imread(file)
        imgStack.append(tmpImg)
        # imgStackShow.append([plt.imshow(tmpImg, vmin=0, vmax=15000, cmap=cm.Greys_r, animated=True)])

    # get stacked numoy array form imgStack
    zStackImg = np.stack(imgStack, axis=-1)
    if method == "max":
        zStackProj = zStackImg.max(axis=2)
    else:
        raise NotImplementedError(f"{method} has not been implemented")
    return zStackProj
