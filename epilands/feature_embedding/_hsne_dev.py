# Import libraries
from __future__ import annotations
from typing import Tuple
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.spatial.distance import squareform, pdist
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import time
import random

# Relative imports
from ..read_write import save_dataframe_to_csv
from ..config import NAME_SEPARATOR_
from ..dev import hmdspy
from ..dev import hsnepy

allRad = list(np.linspace(0.5, 3.5, 10))
numSamples = 100  # randomly sample 100 cell and get the distance matrix for hMDS


# for shapard diagram
def func_powerlaw(x, m, c, y0):
    return c * ((x) ** m) + y0


def generate_hsne(
    df: pd.DataFrame,
    n_components: int = 2,
    save_to: str = None,
    save_fit_data: bool = True,
    **kwargs,
) -> Tuple[hsnepy.HSNE, pd.DataFrame, pd.DataFrame]:
    """
    Function to calculate the multidimensional scaling of a distance matrix.
    The function creates a 2-dimensional representation of the data based on the
    distance matrix.

    Parameters:
    df_pdist (DataFrame): The distance matrix between all of the data points.
    The rows and columns of the matrix should be the same.

    Returns:
    df_mds (DataFrame): A DataFrame containing the two-dimensional
    representation of the data.
    df_mds_params (DataFrame): A DataFrame containing the stress and stress1 (Kruskals)
    of the data, as well as the number of iterations to find the optimal stress.
    """

    # apply hmds to find the best radius
    kappaList = []
    radList = []
    for r in allRad:
        # initialize hmds model
        hmdsModel = hmdspy.HMDS(
            r_max=r, n_components=n_components, max_iter=1000, verbose=False
        )

        for i in range(0, 10):
            # random sample numSamples cells
            sample = random.sample(
                range(0, len(df.index)),
            )
            # get the data of the sample
            tmpData = df[sample, :]  # does this work?
            # get the distance matrix
            tmpDist = squareform(pdist(tmpData))
            # s1=time.time()
            hmdsModel.fit(tmpDist)
            # s2=time.time()
            # print(s2-s1)
            Y_hyper = hmdsModel.embedding_

            Y = hsnepy._rescale_angle(Y_hyper)
            emb_dist = squareform(hsnepy._dist_Hyperbolic(Y))
            org_dist = squareform(tmpDist)

            # fit the power law to evaluate the curvature of the manifold (kappa)
            # shift the values
            dist_thres = 1e-4
            org_dist = org_dist - min(org_dist) + dist_thres
            popt1, pcov1 = curve_fit(
                func_powerlaw,
                org_dist,
                emb_dist,
                p0=[1, 3, 0],
                maxfev=5000,
                bounds=([-np.inf, -np.inf, 0], [np.inf, np.inf, np.nanmin(org_dist)]),
            )

            kappa_val = popt1[0] - 1
            kappaList.append(kappa_val)
            radList.append(r)

            """
            visualization
            ====================
            # #plot the shapard diagram with kappa value
            # plt.figure(figsize=(5, 5))
            # plt.scatter(org_dist, emb_dist, s=1, c='k')
            # #plot line
            # x = np.linspace(0, np.nanmax(org_dist), 100)
            # y = func_powerlaw(x, *popt1)
            # plt.plot(x, y, 'r-', label='fit: m=%5.3f, c=%5.3f, y0=%5.3f, k=%5.3f' % tuple([*popt1,kappa_val]))
            # plt.xlabel('Original Distance')
            # plt.ylabel('Embedded Distance')
            # plt.legend()
            # plt.savefig(savePath+'/shep_'+str(nf)+'_rad'+str(r)+'_dim'+str(dim)+'_rond'+str(i)+'.png')
            # plt.close()  
            ===================
            """

    # fit the line to radList and kappaList
    X = np.array(radList).reshape(-1, 1)
    Y = np.array(kappaList).reshape(-1, 1)
    reg = LinearRegression().fit(X, Y)
    # get x value where y is 0 (Rdata)
    Rdata = -reg.intercept_ / reg.coef_

    """
    visualization
    ====================
    #violin plot for the kappa values with respect to the radius and line fitted to the data
    #saparate the data into different radius
    # dataDict={}
    # for i in range(0,len(radList)):
    #     if radList[i] not in dataDict:
    #         dataDict[radList[i]]=[kappaList[i]]
    #     else:
    #         dataDict[radList[i]].append(kappaList[i])
    # plt.violinplot(dataDict.values(), list(dataDict.keys()),
    #                showmeans=True, showmedians=True)
    # plt.plot(np.linspace(0, 4, 100).reshape(-1, 1), reg.predict(np.linspace(0, 4, 100).reshape(-1, 1)), color='red', linewidth=2)
    # plt.xlabel('Radius')
    # plt.ylabel('Kappa')
    # #save the plot
    # plt.savefig(savePath+'/Kappa_vs_Radius_'+str(nf)+'_dim'+str(dim)+'.png')
    # plt.close()  
    ===================
    """

    # apply hsne with the best radius (Rdata)
    # initialize hmds model
    hsneModel = hsnepy.HSNE(
        n_components=n_components,
        init_R=Rdata,
        early_exaggeration=4,
        learning_rate=1,
        n_iter=10000,
    )
    hsneModel.fit(X=df.to_numpy())

    df_hsne = pd.DataFrame(index=df.index)
    df_hsne[
        ["hSNE{}".format(i + 1) for i in range(n_components)]
    ] = hsneModel.embedding_
    df_hsne.attrs["name"] = NAME_SEPARATOR_.join((df.attrs["name"], "hSNE"))

    if save_fit_data:
        save_dataframe_to_csv(df_hsne, save_to)
