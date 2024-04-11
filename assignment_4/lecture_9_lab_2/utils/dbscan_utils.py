from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
import seaborn as sns
from kneed import KneeLocator
import pandas as pd
import numpy as np
from utils.clustering_utils import find_right_s_for_kneed_elbow
from sklearn.cluster import DBSCAN
import hdbscan.validity as dbcv_hdbscan


def get_best_eps(data, metric='l2', eps_knee_detection_sensitivity=3.0):
    """
    Determine DBSCAN eps and min_samples. Look at the distance from a point to its kth nearest neighbor k-dist. Compute
    the k-dist for all the data points for some k, sort them in increasing order, and then plot the sorted values, we
    expect to see a sharp change at the value of k-dist that corresponds to a suitable value of eps. Select this
    distance as the eps parameter and take the value of k as the min_samples parameter, then points for which k-dist is
    less than eps will be labeled as core points, while other points will be labeled as noise or border points.

    :param data:
    :param metric: 'l1' or 'l2'
    :param eps_knee_detection_sensitivity
    :return:
    """

    # build the kdtree - KDTree for fast generalized N-point problems
    kdt = KDTree(data, metric=metric)

    k_list = [3, 4, 5, 6]
    df_row_dict_list = []
    closest_kth_dict = {}
    for k in k_list:

        # get the distance to the kth closest data object for all data objects
        closest_kth = []
        for i in range(data.shape[0]):
            dist, _ = kdt.query(data[i, :].reshape(1, -1), k+1)  # add 1 - the subject point with distance = 0 included
            closest_kth.append(dist[0, -1])

        # sort the distances and store in dictionary
        closest_kth_dict[k] = sorted(closest_kth)

        # set the kneed parameters to find elbow
        curve = 'convex'
        direction = 'increasing'
        drop_first_percent = 0.10
        drop_first = int(drop_first_percent * data.shape[0])
        y = closest_kth_dict[k][drop_first:]  # drop first 100 to drop first concave knee

        # find the right S value - only run to find right S
        find_right_s = False
        if find_right_s:
            find_right_s_for_kneed_elbow(None, y, curve, direction)

        # find the elbow and get eps
        s = eps_knee_detection_sensitivity  # found using tool by setting find_right_s above to True
        kneedle = KneeLocator(list(range(len(y))), y, S=s, curve=curve, direction=direction)
        idx = kneedle.elbow
        eps = y[idx]
        df_row_dict_list.append(
            {
                'index': drop_first + idx,
                'k': k,
                'eps': eps
            }
        )

    print('\neps determined from behavior of the distance from a point to its kth nearest neighbor')
    print('   sort distance, graph and get distance at the knee')
    temp_df = pd.DataFrame(df_row_dict_list).sort_values('eps')
    print('\n', temp_df, sep='')

    # plot the curves used to determine eps for each k
    for k in k_list:
        sns.lineplot(x=range(len(closest_kth_dict[k])), y=closest_kth_dict[k], label=k)
    plt.legend()
    plt.grid()
    plt.show()

    return temp_df


def get_dbscan_masks(dbscan):

    """
    Returns boolean masks that can be used to select out data objects

    :param dbscan:
    :return:
    """

    # return an array of boolean False with the same shape and type as dbscan.labels_
    core_mask = np.zeros_like(dbscan.labels_, dtype=bool)

    # modify the boolean False array to True for all core data objects
    core_mask[dbscan.core_sample_indices_] = True

    # create a boolean mask indicating all anomalous (non-core data objects)
    anomalies_mask = dbscan.labels_ == -1

    # use core_data_object_mask and anomalies_data_object_mask to make a mask indicating non-core data objects - these
    # are border data objects
    border_mask = ~(core_mask | anomalies_mask)

    # use the border_data_objects_mask and the core_data_object_mask to make a core_and_border_data_object_mask
    core_and_border_mask = (core_mask | border_mask)

    assert(core_mask.sum() + anomalies_mask.sum() + border_mask.sum() == dbscan.labels_.shape[0])
    assert (dbscan.labels_.shape[0] - core_and_border_mask.sum() == anomalies_mask.sum())

    return border_mask, anomalies_mask, border_mask, core_and_border_mask, core_mask


def plot_dbscan(dbscan, cap_x, show_x_labels=True, show_y_labels=True, extra_title=None):

    border_mask, anomalies_mask, border_mask, _, core_mask = get_dbscan_masks(dbscan)

    cores = dbscan.components_
    anomalies = cap_x[anomalies_mask]
    non_cores = cap_x[border_mask]

    if cores.shape[0] > 0:
        plt.scatter(cores[:, 0], cores[:, 1], marker='*', s=20, c=dbscan.labels_[core_mask], label='core')

    if anomalies.shape[0] > 0:
        plt.scatter(anomalies[:, 0], anomalies[:, 1], c="r", marker="x", s=100, label='anomalies')

    if non_cores.shape[0] > 0:
        plt.scatter(non_cores[:, 0], non_cores[:, 1], c=dbscan.labels_[border_mask], marker=".", label='border')

    if show_x_labels:
        plt.xlabel("$x_1$")
    else:
        plt.tick_params(labelbottom=False)
    if show_y_labels:
        plt.ylabel("$x_2$", rotation=0)
    else:
        plt.tick_params(labelleft=False)

    title = f"eps={dbscan.eps:.2f}, min_samples={dbscan.min_samples}\nlegend identifies dbscan data object types by " \
            f"shape"
    if extra_title is not None:
        title = title + extra_title
    plt.title(title)

    plt.grid()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))


def plot_data(cap_x):
    plt.plot(cap_x[:, 0], cap_x[:, 1], 'k.', markersize=2)


def plot_centroids(centroids, weights=None, circle_color='w', cross_color='k'):

    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]

    plt.scatter(centroids[:, 0], centroids[:, 1], marker='o', s=35, linewidths=8, color=circle_color, zorder=10,
                alpha=0.9)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=2, linewidths=12, color=cross_color, zorder=11, alpha=1)


def plot_decision_boundaries(clusterer, cap_x, resolution=1000, show_centroids=True, show_x_labels=True,
                             show_y_labels=True):

    min_vals = cap_x.min(axis=0) - 0.1
    max_vals = cap_x.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(min_vals[0], max_vals[0], resolution),
                         np.linspace(min_vals[1], max_vals[1], resolution))
    cap_z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    cap_z = cap_z.reshape(xx.shape)

    plt.contourf(cap_z, extent=(min_vals[0], max_vals[0], min_vals[1], max_vals[1]), cmap="Pastel2")
    plt.contour(cap_z, extent=(min_vals[0], max_vals[0], min_vals[1], max_vals[1]), linewidths=1, colors='k')

    plot_data(cap_x)
    if show_centroids:
        plot_centroids(clusterer.cluster_centers_)

    if show_x_labels:
        plt.xlabel("$x_1$")
    else:
        plt.tick_params(labelbottom=False)
    if show_y_labels:
        plt.ylabel("$x_2$", rotation=0)
    else:
        plt.tick_params(labelleft=False)


def illustrative_plots_for_selecting_the_best_fitted_dbscan(results_df, idx_max=None):

    print(results_df[['k_dist_eps', 'f_eps', 'min_samples', 'n_clusters', 'dbcv_score']])
    sns.scatterplot(data=results_df, x='n_clusters', y='dbcv_score')
    plt.xlabel('number of clusters found')
    plt.ylabel('validity index (DBCV)')
    plt.title('')
    plt.grid()
    plt.show()

    sns.scatterplot(data=results_df, x='f_eps', y='dbcv_score')
    for x, y, label in zip(results_df.f_eps, results_df.dbcv_score, results_df.n_clusters):
        plt.text(x, y, label)
    plt.axvline(x=results_df.k_dist_eps.values[0], c='r')
    extra_title = ''
    if idx_max is not None:
        plt.axvline(x=results_df.f_eps[idx_max], c='k')
        extra_title = '\nblack vertical line = best eps from eps scan'
    plt.xlabel('eps')
    plt.ylabel('validity index (DBCV)')
    plt.title(f'data point labels = numbers of clusters found\nred vertical line = eps from k-distance plot '
              f'{extra_title}')
    plt.grid()
    plt.show()


def select_the_best_fitted_dbscan(results_df):
    """
    this algo was developed on the moons data set with varying noise levels

    :param results_df:
    :return:
    """

    # setting algo_dev to True will display algo development tools
    algo_dev = False

    if algo_dev:
        illustrative_plots_for_selecting_the_best_fitted_dbscan(results_df)

    # get the number of clusters that corresponds to the lowest dbcv_score
    idx_min = results_df.dbcv_score.idxmin()
    n_clusters_max = results_df.n_clusters[idx_min]

    # only keep rows with that number of clusters or fewer then pick row with largest dbcv_score
    results_df = results_df[results_df.n_clusters <= n_clusters_max]
    idx_max = results_df.dbcv_score.idxmax()

    best_fitted_dbscan = results_df.fitted_dbscan[idx_max]

    if algo_dev:
        illustrative_plots_for_selecting_the_best_fitted_dbscan(results_df, idx_max)

    return best_fitted_dbscan


def perform_dbscan_clustering(cap_x, y=None, eps_knee_detection_sensitivity=3.0, eps_scan_range=(0.5, 1.8, 0.1),
                              metric='euclidean'):

    eps_k_df = get_best_eps(cap_x, eps_knee_detection_sensitivity=eps_knee_detection_sensitivity)
    eps = eps_k_df.eps.values.max()
    min_samples = eps_k_df.loc[eps_k_df.eps == eps, 'k'].values[0]

    df_row_dict_list = []
    for factor in np.arange(eps_scan_range[0], eps_scan_range[1], eps_scan_range[2]):

        f_eps = factor * eps

        dbscan = DBSCAN(
            eps=f_eps,
            min_samples=min_samples,
            metric=metric,
            metric_params=None,
            algorithm='auto',
            leaf_size=30,
            p=None,
            n_jobs=None
        )

        dbscan.fit(cap_x)

        clusters = np.unique(dbscan.labels_)
        n_clusters = clusters[clusters != -1].shape[0]

        if metric == 'manhattan':
            metric = 'cityblock'

        try:
            dbcv_score = dbcv_hdbscan.validity_index(cap_x.astype('double'), dbscan.labels_, metric=metric)
        except ValueError as e:
            print(e)
            dbcv_score = np.nan

        df_row_dict_list.append(
            {
                'k_dist_eps': eps,
                'f_eps': f_eps,
                'min_samples': min_samples,
                'n_clusters': n_clusters,
                'dbcv_score': dbcv_score,
                'fitted_dbscan': dbscan
            }
        )

    # assemble results and drop rows with nan in dbcv_score
    results_df = pd.DataFrame(df_row_dict_list)
    results_df = results_df.dropna(subset=['dbcv_score'])

    # get the best fitted dbscan
    best_fitted_dbscan = select_the_best_fitted_dbscan(results_df)

    return {
        'cap_x': cap_x,
        'y': y,
        'fitted_dbscan': best_fitted_dbscan
    }


if __name__ == "__main__":
    pass
