from __future__ import annotations

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Relative imports
from ..read_write import save_dataframe_to_csv
from ..tools import join_iterable, create_group_col


def calculate_1D_MIEL_distance(
    df_pdist: pd.DataFrame,
    reference_group: tuple,
    group_col_list: list,
):
    if isinstance(df_pdist.index, pd.MultiIndex):
        df_workinginfo = df_pdist.groupby(group_col_list)
    else:
        raise ValueError(
            "generate_MIEL_distances: df_pdist must have a multiindex in order to calculate MIEL distances (reference groups needed)"
        )

    X = df_workinginfo.get_group(reference_group).mean()
    # X.attrs['name'] = f'Distance From {join_iterable(reference_group)}'
    X.attrs["name"] = f"MIEL_distance_from_{join_iterable(reference_group)}"
    df_MIEL_distances = extract_group_and_meta_data(
        df_pdist, group_col_list, group_col_name="Group"
    )
    df_MIEL_distances.loc[:, X.attrs["name"]] = X
    df_MIEL_distances.attrs["name"] = df_pdist.attrs["name"] + "_MIELdist1D"
    return df_MIEL_distances


def generate_MIEL_distances(
    df_pdist: pd.DataFrame,
    reference_group_A: tuple,
    reference_group_B: tuple,
    group_col_list: list,
    save_to: str,
    save_fit_data: bool = True,
    **kwargs,
):
    """
    generate_MIEL_distances

    Parameters
    ----------
    df_pdist : dataframe
        dataframe of distances.

    reference_group_A : tuple(string1, string2, ... stringN) where N = len(group_col_list)
        name of left reference group. Must be a unique value contained in the groups geated by df_pdist.groupby(group_col_list)

    reference_group_B : tuple(string1, string2, ... stringN) where N = len(group_col_list)
        name of right reference group. Must be a unique value contained in the groups geated by df_pdist.groupby(group_col_list)

    group_col_list : list
        list of column names that identify groups. Passed to df_pdist.groupby(group_col_list)

    Returns
    -------
    df_MIEL_distances : dataframe
        pandas dataframe of distances transformed via a linear transformation forming the MIEL/miBioAge axis
    """

    if isinstance(df_pdist.index, pd.MultiIndex):
        df_workinginfo = df_pdist.groupby(group_col_list)
    else:
        raise ValueError(
            "generate_MIEL_distances: df_pdist must have a multiindex in order to calculate MIEL distances (reference groups needed)"
        )
    # create the series X and Y which compose the X and Y coordinates of each sample in the new coordinate system (X = distance from reference_group_A, Y = distance from reference_group_B)
    # NOTE: this can be confusing... it helps to notice that the data with LOW X VALUES and HIGH Y VALUES (upper left) cooresponds to the REFERENCE_GROUP_A (FARTHER FROM B, CLOSER TO A)
    # likewise, HIGH X VALUES and LOW Y VALUES (lower right) cooresponds to the REFERENCE_GROUP_B (CLOSER TO B, FARTHER FROM A)

    X = df_workinginfo.get_group(reference_group_A).mean()
    X.attrs["name"] = "X (Distance From {})".format(join_iterable(reference_group_A))
    Y = df_workinginfo.get_group(reference_group_B).mean()
    Y.attrs["name"] = "Y (Distance From {})".format(join_iterable(reference_group_B))
    plt.figure()
    plt.scatter(x=X, y=Y)
    plt.title("X vs Y")
    plt.xlabel(X.attrs["name"])
    plt.ylabel(Y.attrs["name"])
    plt.axis("square")
    # plt.show()
    plt.close()
    print("X, Y successfully computed")

    # Calculate centroid of young for transposition to (0,0)
    center_reference_group_A = [
        X.groupby(group_col_list).get_group(reference_group_A).mean(),
        Y.groupby(group_col_list).get_group(reference_group_A).mean(),
    ]
    print(
        "center " + join_iterable(reference_group_A) + " = ", center_reference_group_A
    )
    # Transpose axes such that the young group is at the center
    X = X.map(lambda x: x - center_reference_group_A[0])
    Y = Y.map(lambda y: y - center_reference_group_A[1])

    # Calculate centroid of old post-transpose
    center_reference_group_B = [
        X.groupby(group_col_list).get_group(reference_group_B).mean(),
        Y.groupby(group_col_list).get_group(reference_group_B).mean(),
    ]
    print(
        "center " + join_iterable(reference_group_B) + " = ", center_reference_group_B
    )

    # Calculate basis vectors for new coordinate system
    Xbasis = np.array(
        center_reference_group_B
    )  # create a vector from the centroid of ref.group.A (0,0) to ref.group.B
    Xbasis /= np.linalg.norm(Xbasis)  # normalize it (length = 1)

    Ybasis = np.array(
        [-1 * Xbasis[1], Xbasis[0]]
    )  # create orthogonal vector -- flip the X and Y coordinates, make Y negative
    # Just to be sure:
    Ybasis -= Ybasis.dot(Xbasis) * Xbasis  # make it orthogonal to Xbasis
    Ybasis /= np.linalg.norm(Ybasis)  # normalize it

    print("Basis Vectors successfully found")
    print("To confirm, dot product is:", Xbasis.dot(Ybasis))

    # construct a transformation matrix and an array of X,Y points for transformation
    transformation_matrix = np.array([[Xbasis[0], Ybasis[0]], [Xbasis[1], Ybasis[1]]])
    points = np.array([list(X), list(Y)])

    print("Transformation matrix complete:", transformation_matrix)

    # Transform data to put the centroid of the old group on the X axis
    new_points = np.linalg.inv(transformation_matrix).dot(points)

    print("Success! Linear transformation of the data completed")

    Xprime = pd.Series(new_points[0], index=X.index, name="MIEL Distance")
    Yprime = pd.Series(new_points[1], index=Y.index, name="MIEL Distance Orthogonal")
    plt.figure()
    plt.scatter(x=Xprime, y=Yprime)
    plt.title("MIEL Distance vs MIEL Distance Orthogonal")
    plt.xlabel(Xprime.name)
    plt.ylabel(Yprime.name)
    plt.axis("square")
    # plt.show()
    plt.close()

    print("Data successfully transformed:")
    df_MIEL_distances = extract_group_and_meta_data(
        df_pdist, group_col_list, group_col_name="Group"
    )
    df_MIEL_distances.loc[:, Xprime.name] = Xprime
    df_MIEL_distances.loc[:, Yprime.name] = Yprime

    df_MIEL_distances.sort_index(inplace=True)
    df_MIEL_distances.attrs["name"] = df_pdist.attrs["name"] + "_MIELdist"

    # order = df_MIEL_distances.Group.str.len().sort_values().index
    # df_MIEL_distances_1 = df_MIEL_distances.reindex(order)
    # df_MIEL_distances_1.attrs['name'] = df_MIEL_distances.attrs['name']
    # df_MIEL_distances = df_MIEL_distances_1
    # df_MIEL_distances
    # df_MIEL_distances_groupby_sample.index.attrs['name'] = 'index'
    if save_fit_data:
        save_dataframe_to_csv(df_MIEL_distances, save_to, **kwargs)
    return df_MIEL_distances


def extract_group_and_meta_data(
    df, group_col_list, group_col_name, metadata_cols: str = None, **kwargs
):
    df_work = pd.DataFrame(index=df.index)
    df_work.loc[:, group_col_name] = create_group_col(
        df, group_col_list, group_col_name
    )
    for i in group_col_list:
        if i in df.columns:
            df_work.loc[:, i] = df[i]
    print(metadata_cols)
    if metadata_cols != None:
        df_work.loc[:, metadata_cols] = df[metadata_cols]
    df_work.attrs["name"] = df.attrs["name"]
    return df_work
