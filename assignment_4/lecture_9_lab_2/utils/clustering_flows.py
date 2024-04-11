import pandas as pd
import numpy as np
import utils.k_means_utils as kmu
import utils.dbscan_utils as dbsu
import utils.clustering_utils as clu
import seaborn as sns
import matplotlib.pyplot as plt


def plot_the_dbscan_clustering_results(cap_x, dbscan_results_dict, data_set_name=None):

    if data_set_name is None:
        data_set_name = ''

    dbscan = dbscan_results_dict['fitted_dbscan']
    extra_title = f'\ndata_set_name: {data_set_name}'
    dbsu.plot_dbscan(dbscan, cap_x, extra_title=extra_title)
    plt.show()


def plot_the_data_sets(cap_x, y=None, data_set_name=None):

    if data_set_name is None:
        data_set_name = ''

    if y is not None:
        # plot the data with truth labels
        ax = sns.scatterplot(x=cap_x[:, 0], y=cap_x[:, 1], hue=y)
        ax.set_aspect('equal', adjustable='box')
        plt.title(f'data_set_name: {data_set_name} - with truth labels')
        plt.grid()
        plt.show()

    # plot the data without the truth labels
    ax = sns.scatterplot(x=cap_x[:, 0], y=cap_x[:, 1])
    ax.set_aspect('equal', adjustable='box')
    plt.title(f'data_set_name: {data_set_name}\nwithout truth labels')
    plt.grid()
    plt.show()


def plot_the_k_means_clustering_results(cap_x, k_means_results_dict, data_set_name=None):

    if data_set_name is None:
        data_set_name = ''

    k_means_n_clusters = k_means_results_dict['n_clusters']
    k_means = k_means_results_dict['fitted_k_means_dict'][k_means_n_clusters]

    ax = sns.scatterplot(x=cap_x[:, 0], y=cap_x[:, 1], hue=k_means.labels_)
    plt.scatter(k_means.cluster_centers_[:, 0], k_means.cluster_centers_[:, 1], marker='+', s=200, linewidths=3,
                c='r')
    ax.set_aspect('equal', adjustable='box')
    plt.title(f'data_set_name: {data_set_name}\nbest k-means solution')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid()
    plt.show()


def clustering_flow_1(cap_x, n_clusters_list, df_row_dict_list, y=None, data_set_name=None,
                      find_right_k_means_elbow_sens=False, k_means_elbow_sens=1.0, k_means_min_elbow_slope_diff=30,
                      enhanced_k_means=False, metric='euclidean'):

    # take care of a few things
    if y is not None:
        true_n_clusters = np.unique(y).shape[0]
    else:
        true_n_clusters = None

    # plot the data set if 2D
    if cap_x.shape[1] == 2:
        plot_the_data_sets(cap_x, y, data_set_name)

    # measure tendency to cluster
    cap_h = clu.get_hopkins_statistic(cap_x, metric, plot=True)

    # perform k-means clustering
    k_means_results_dict = \
        kmu.perform_k_means_clustering(cap_x, n_clusters_list, y=y, init='random',
                                       min_elbow_slope_diff=k_means_min_elbow_slope_diff,
                                       find_right_sensitivity=find_right_k_means_elbow_sens,
                                       sensitivity=k_means_elbow_sens, enhanced_k_means=enhanced_k_means)
    k_means_n_clusters = k_means_results_dict['n_clusters']
    k_means = k_means_results_dict['fitted_k_means_dict'][k_means_n_clusters]

    temp_metric = metric
    if metric == 'manhattan':
        metric = 'cityblock'
    k_means_internal_indices_dict = \
        clu.get_internal_indices(cap_x, k_means.labels_, type_of_clustering='prototype_based', metric=metric)
    metric = temp_metric

    # plot the k-means solution if data set is 2D
    if cap_x.shape[1] == 2:
        plot_the_k_means_clustering_results(cap_x, k_means_results_dict, data_set_name)

    if k_means_results_dict['slope_change_at_elbow_test_results'] == 'pass':

        # collect k_means results
        df_row_dict_list.append(
            {
                'data_set_name': data_set_name,
                'algo': 'k_means',
                'slope_change_at_elbow_test_results': k_means_results_dict['slope_change_at_elbow_test_results'],
                'elbow_slope_diff': k_means_results_dict['elbow_slope_diff'],
                'min_elbow_slope_diff': k_means_results_dict['min_elbow_slope_diff'],
                'n_clusters_found': k_means_n_clusters,
                'silhouette_score': k_means_internal_indices_dict['silhouette_score'],
                'true_n_clusters': true_n_clusters,
                'hopkins_statistic': cap_h,
                'cluster_labels': k_means.labels_
            }
        )

    elif k_means_results_dict['slope_change_at_elbow_test_results'] == 'fail':

        slope_change_at_elbow_test_results = k_means_results_dict['slope_change_at_elbow_test_results']
        print(f'\nk-means: {slope_change_at_elbow_test_results} - try dbscan')

        # perform dbscan clustering
        dbscan_results_dict = dbsu.perform_dbscan_clustering(cap_x, y=y, metric=metric)
        dbscan = dbscan_results_dict['fitted_dbscan']
        clusters = np.unique(dbscan.labels_)
        dbscan_n_clusters = clusters[clusters != -1].shape[0]

        # plot the dbscan solution if data set is 2D
        if cap_x.shape[1] == 2:
            plot_the_dbscan_clustering_results(cap_x, dbscan_results_dict, data_set_name)

        # get internal indices for the dbscan clustering - print them out
        dbscan_internal_indices_dict = clu.get_internal_indices(cap_x.astype('double'), dbscan.labels_,
                                                                type_of_clustering='density_based', metric=metric)

        print(f'dbscan_n_clusters found: {dbscan_n_clusters}')
        validity_index = dbscan_internal_indices_dict['validity_index']
        print(f'validity_index: {validity_index}')

        # collect dbscan results
        df_row_dict_list.append(
            {
                'data_set_name': data_set_name,
                'algo': 'dbscan',
                'eps': dbscan.eps,
                'min_samples': dbscan.min_samples,
                'n_clusters_found': dbscan_n_clusters,
                'validity_index': dbscan_internal_indices_dict['validity_index'],
                'true_n_clusters': true_n_clusters,
                'hopkins_statistic': cap_h,
                'cluster_labels': dbscan.labels_
            }
        )

    return df_row_dict_list


def clustering_flow_1_return_dict(cap_x, n_clusters_list, y=None, data_set_name=None,
                                  find_right_k_means_elbow_sens=False, k_means_elbow_sens=1.0,
                                  k_means_min_elbow_slope_diff=30, enhanced_k_means=False, metric='euclidean'):

    # wrapper for clustering_flow_1() - used to deliver a dictionary instead of a list

    df_row_dict_list = []
    df_row_dict_list = clustering_flow_1(
        cap_x,
        n_clusters_list,
        df_row_dict_list,
        y=y,
        data_set_name=data_set_name,
        find_right_k_means_elbow_sens=find_right_k_means_elbow_sens,
        k_means_elbow_sens=k_means_elbow_sens,
        k_means_min_elbow_slope_diff=k_means_min_elbow_slope_diff,
        enhanced_k_means=enhanced_k_means,
        metric=metric
    )

    df_row_dict = df_row_dict_list[0]

    return {
        'df_row_dict': df_row_dict
    }


def get_number_of_clusters(cap_x_y_df):

    if 'y' in cap_x_y_df.columns:
        n_clusters = cap_x_y_df.y.nunique()
    else:
        n_clusters = None

    return n_clusters


def clustering_flow_1_test_1(data_set_dicts, find_right_k_means_elbow_sens=False, k_means_elbow_sens=1.0,
                             k_means_min_elbow_slope_diff=30, enhanced_k_means=False, metric='euclidean'):

    n_clusters_list = list(range(2, 16))
    df_row_dict_list = []
    for data_set_name, data_set_dict in data_set_dicts.items():

        cap_x = data_set_dict['cap_x']
        y = data_set_dict['y']

        return_dict = \
            clustering_flow_1_return_dict(
                cap_x,
                n_clusters_list,
                y=y,
                data_set_name=data_set_name,
                find_right_k_means_elbow_sens=find_right_k_means_elbow_sens,
                k_means_elbow_sens=k_means_elbow_sens,
                k_means_min_elbow_slope_diff=k_means_min_elbow_slope_diff,
                enhanced_k_means=enhanced_k_means,
                metric=metric
            )
        df_row_dict = return_dict['df_row_dict']
        df_row_dict['data_set_type'] = data_set_dict['data_set_type']
        df_row_dict['spec_type'] = data_set_dict['spec_type']
        df_row_dict['spec_value'] = data_set_dict['spec_value']
        df_row_dict_list.append(df_row_dict)

    results_df = pd.DataFrame(df_row_dict_list).sort_values(['algo', 'data_set_type', 'spec_value'])

    return results_df


if __name__ == "__main__":
    pass
