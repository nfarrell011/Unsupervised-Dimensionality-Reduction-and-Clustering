import sys
import matplotlib.pyplot as plt
import seaborn as sns
from kneed import KneeLocator
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
import hdbscan.validity as dbcv_hdbscan
from scipy.stats import linregress
import numpy as np
from sklearn.neighbors import KDTree


def find_right_s_for_kneed_elbow(x, y, curve, direction):
    print('\ninside find_right_s_for_kneed_elbow() trying to find right sensitivity for elbow detection')
    print('human in the loop - will terminate at end of function')
    print('get best sensitivity')

    # if we only have a y series - make x-axis
    if x is None:
        x = range(len(y))

    # plot the data we are trying to find knee on
    sns.lineplot(x=x, y=y)
    plt.grid()
    plt.show()

    # scan over S
    elbows_dict = {}
    norm_elbows = []
    sensitivity = [0.01, 0.1, 1, 3, 5, 7, 10, 100, 200]
    sensitivity_drop_list = []
    kneedle = None
    for s in sensitivity:
        kneedle = KneeLocator(x, y, S=s, curve=curve, direction=direction)

        if kneedle.norm_elbow is None:
            sensitivity_drop_list.append(s)
            continue

        elbows_dict[s] = kneedle.elbow
        norm_elbows.append(kneedle.norm_elbow)

    # remove sensitivities
    print(f'\nthese sensitivities did not produce an elbow:\n{sensitivity_drop_list}')
    for s in sensitivity_drop_list:
        sensitivity.remove(s)

    # print some results
    print(f'\nelbows_dict - key = sensitivity, value = number of clusters:\n{elbows_dict}')

    # generate plot
    plt.figure(figsize=(8, 6))
    plt.plot(kneedle.x_normalized, kneedle.y_normalized)
    plt.plot(kneedle.x_difference, kneedle.y_difference)
    colors = ["r", "g", "k", "m", "c", "orange"]
    for k, c, s in zip(norm_elbows, colors, sensitivity):
        plt.vlines(k, 0, 1, linestyles="--", colors=c, label=f"S = {s}")

    plt.grid()
    plt.legend()
    plt.show()

    sys.exit(f'\nwhen find_right_sensitivity is True the algorithm runs and terminates here - set '
             f'find_right_sensitivity to False for the script to run normally')


def locate_elbow_in_inertia_vs_n_clusters_curve(k_vs_inertia_df, find_right_sensitivity=False, sensitivity=1.0):

    """

    :param k_vs_inertia_df: data frame with n_clusters and inertia attributes
    :param find_right_sensitivity: if True an application runs that helps find the right sensitivity - script stops
    :param sensitivity: the sensitivity used to locate the elbow in the inertia vs number of clusters curve generated
                        with k-means
    :return: n_clusters at elbow location
    """

    # set the kneed parameters to find elbow in inertia vs n_clusters curve
    curve = 'convex'
    direction = 'decreasing'

    # find the right S value - only run to find right S for the inertia vs n_clusters curve
    if find_right_sensitivity:
        find_right_s_for_kneed_elbow(k_vs_inertia_df.n_clusters, k_vs_inertia_df.inertia, curve, direction)

    # find the elbow in inertia vs n_clusters curve
    kneedle = KneeLocator(k_vs_inertia_df.n_clusters, k_vs_inertia_df.inertia, S=sensitivity, curve=curve,
                          direction=direction)
    try:
        print(f'\nelbow in the inertia curve is located at {kneedle.elbow}')
        n_clusters = kneedle.elbow
    except Exception as e:
        print(e)
        print(f'\ncould not find elbow in the inertia curve\n')
        n_clusters = None

    return n_clusters


def get_interior_points_of_curve(results_df, x='n_clusters', y='inertia'):
    results_df = results_df.iloc[1:, :].iloc[:-1, :]
    interior_x_values = results_df[x].values
    interior_y_values = results_df[y].values
    return interior_x_values, interior_y_values


def get_slope_change_all_interior_n_clusters(results_df, print_out=False):

    interior_x_values, _ = get_interior_points_of_curve(results_df)

    if print_out:
        print(f'\nlook at the slope change of the inertia vs number of clusters curve:')

    elbow_slope_diff_list = [np.nan]
    for n_clusters in interior_x_values:
        _, elbow_slope_diff = test_slope_change_at_elbow(results_df, n_clusters, print_out=False)
        elbow_slope_diff_list.append(elbow_slope_diff)

        if print_out:
            print(f'   n_clusters: {n_clusters}, elbow_slope_diff: {elbow_slope_diff}')

    elbow_slope_diff_list.append(np.nan)
    results_df['elbow_slope_diff'] = elbow_slope_diff_list

    return results_df


def test_slope_change_at_elbow(results_df, n_clusters, min_elbow_slope_diff=30, print_out=True):

    # get the points before, at, and after elbow
    n_clusters_m1 = results_df.loc[results_df.n_clusters == n_clusters - 1, :].values.flatten()
    n_clusters_at_elbow = results_df.loc[results_df.n_clusters == n_clusters, :].values.flatten()
    n_clusters_p1 = results_df.loc[results_df.n_clusters == n_clusters + 1, :].values.flatten()

    # calc the slope before elbow
    slope_before_elbow = linregress(
        [n_clusters_m1[0], n_clusters_at_elbow[0]],
        [n_clusters_m1[1], n_clusters_at_elbow[1]]
    ).slope

    # calc the slope after elbow
    slope_after_elbow = linregress(
        [n_clusters_at_elbow[0], n_clusters_p1[0]],
        [n_clusters_at_elbow[1], n_clusters_p1[1]]
    ).slope

    # calc the slope difference
    elbow_slope_diff = slope_after_elbow - slope_before_elbow

    # test the slope difference
    if elbow_slope_diff < min_elbow_slope_diff:
        slope_change_at_elbow_test_results = 'fail'
    else:
        slope_change_at_elbow_test_results = 'pass'

    if print_out:
        print(f'\nelbow_slope_diff: {elbow_slope_diff}')
        print(f'\nmin_elbow_slope_diff: {min_elbow_slope_diff}')
        print(f'\nslope_change_at_elbow_test_results: {slope_change_at_elbow_test_results}')

    return slope_change_at_elbow_test_results, elbow_slope_diff


def get_unsupervised_internal_indices_list(type_of_clustering=None):

    if type_of_clustering is None:
        unsupervised_metric_list = [
            calinski_harabasz_score,
            davies_bouldin_score,
            silhouette_score,
            dbcv_hdbscan.validity_index
        ]
    elif type_of_clustering == 'density_based':
        # density
        unsupervised_metric_list = [
            dbcv_hdbscan.validity_index
        ]
    elif type_of_clustering == 'prototype_based':
        # cohesion and separation
        unsupervised_metric_list = [
            calinski_harabasz_score,
            davies_bouldin_score,
            silhouette_score,
        ]
    else:
        sys.exit(f'\ntype_of_clustering {type_of_clustering} is not recognized.')

    return unsupervised_metric_list


def get_metric_name_string_from_instantiated_object(metric):
    metric_name_string = str(metric).split(' ')[1]
    return metric_name_string


def get_internal_indices(cap_x, labels_pred, type_of_clustering=None, metric='euclidean'):

    unsupervised_internal_indices_list = get_unsupervised_internal_indices_list(type_of_clustering)

    # get the metrics
    results_dict = {}
    for internal_indices in unsupervised_internal_indices_list:

        metric_name_string = get_metric_name_string_from_instantiated_object(internal_indices)
        if metric_name_string == 'silhouette_score':
            if metric == 'manhattan':
                metric = 'cityblock'
            metric_value = internal_indices(cap_x, labels=labels_pred, metric=metric)
        else:
            metric_value = internal_indices(cap_x, labels=labels_pred)

        # round the results
        try:
            results_dict[metric_name_string] = round(metric_value, 4)
        except TypeError:
            results_dict[metric_name_string] = metric_value.round(4)

    return results_dict


def get_nearest_neighbor_distance(cap_x, metric='euclidean'):

    # build the kdtree
    kdt = KDTree(cap_x, metric=metric)

    # get list of nearest neighbors
    nn_dist_list = []
    for i in range(cap_x.shape[0]):
        dist, _ = kdt.query(cap_x[i, :].reshape(1, -1), 2)
        nn_dist_list.append(dist[0, -1])

    return nn_dist_list


def get_randomly_distributed_data(cap_x, seed=42, plots=False):

    data_max = cap_x.max(axis=0)
    data_min = cap_x.min(axis=0)

    np.random.seed(seed)

    randomly_distributed_data = np.random.uniform(low=data_min[0], high=data_max[0], size=(cap_x.shape[0], 1))
    for i in range(cap_x.shape[1] - 1):
        rand_i_dim = np.random.uniform(low=data_min[0], high=data_max[0], size=(cap_x.shape[0], 1))
        randomly_distributed_data = np.concatenate((randomly_distributed_data, rand_i_dim), axis=1)

    if plots and randomly_distributed_data.shape[1] == 2:
        print('\nnoise data with same number of observations')
        sns.scatterplot(x=randomly_distributed_data[:, 0], y=randomly_distributed_data[:, 1])
        plt.grid()
        plt.show()
    else:
        if plots:
            print(f'\nrandomly_distributed_data.shape[1] = {randomly_distributed_data.shape[1]} > 2 - '
                  f'no plot generated\n')

    return randomly_distributed_data


def hopkins_statistic_plot_helper(cap_x, data_set_name, plt_title):

    if cap_x.shape[1] == 2:
        sns.scatterplot(x=cap_x[:, 0], y=cap_x[:, 1])
        plt.title(plt_title)
        plt.grid()
        plt.show()
    else:
        print(f'   {data_set_name} dimensionality = {cap_x.shape[1]} > 2 - no plot generated')


def get_hopkins_statistic(cap_x, metric, plot=False):

    print(f'\ncheck out the clustering tendency of the data set')

    if plot:
        hopkins_statistic_plot_helper(cap_x, 'data to be clustered', 'data to be clustered')

    randomly_distributed_data = get_randomly_distributed_data(cap_x, seed=42)

    if plot:
        plt_title = 'randomly generated data with the same number of data objects\nused to measure clustering tendency'
        hopkins_statistic_plot_helper(randomly_distributed_data, 'randomly generated data', plt_title)

    cap_x_nn_dist_list = get_nearest_neighbor_distance(cap_x, metric)
    randomly_distributed_data_nn_dist_list = get_nearest_neighbor_distance(randomly_distributed_data, metric)
    cap_h = sum(cap_x_nn_dist_list) / (sum(randomly_distributed_data_nn_dist_list) + sum(cap_x_nn_dist_list))
    cap_h = round(cap_h, 3)

    print(f'hopkins_statistic: {cap_h}')

    return cap_h


if __name__ == "__main__":
    pass
