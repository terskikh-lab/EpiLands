# EpiLands -- Epigenetic Landscape Image Analysis in Python

EpiLands is a libary / toolkit for analyzing the epigenetic information contained in microscopic images at a single-nucleus resolution. In its current form, this package provides image preprocessing, segmentation, feature extraction, dimensionality reduction, and visualization. 

If you're interested in EpiLands, check out helper packages [MultiProcessTools](https://github.com/sbp-terskikh-lab/MultiProcessTools) and [MultiProcessHelper](https://github.com/sbp-terskikh-lab/MultiProcessTools) for parallel processing capabilities to speed up segmentation, feature extraction and analysis.

EpiLands as it stands right now is a loosely held together set of functions which aim to provide a lightwieght but useful toolkit for the Terskikh lab, and we are releasing it publically here for anyone attempting to recreate the Terskikh lab's published results. As our current manuscript is under review, the tools provided here may be changed and updated to reflect most accurately what we end up publishing. Additionally, many of the tools provided here are poorly written, bulky, or completely unnecessary and our efforts post-publication will focus on cleaning up this library and removing functions which provide no clear benefit to our enduser.

This project is in its infancy and much of the code is actively being developed (and deleted), so expect major refactoring until future versions specify otherwise. 

EpiLands was developed as part of the Terskih Lab at SBP Medical Discovery Institute