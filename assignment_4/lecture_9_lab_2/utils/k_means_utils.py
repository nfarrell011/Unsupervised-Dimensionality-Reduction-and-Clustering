from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import utils.clustering_utils as clu
import numpy as np


def perform_k_means_clustering(cap_x, n_clusters_list, y=None, init='k-means++', n_init='auto',
                               min_elbow_slope_diff=30, find_right_sensitivity=False, sensitivity=1.0,
                               enhanced_k_means=False):

    # fit k-means for each value of n_clusters in n_clusters_list - store results in data frame
    fitted_k_means_dict = {}
    df_row_dict_list = []
    for n_clusters in n_clusters_list:

        k_means = KMeans(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=300,
            tol=0.0001,
            verbose=0,
            random_state=None,
            copy_x=True,
            algorithm='lloyd'
        )

        k_means.fit_predict(cap_x)

        k_means_internal_indices_dict = \
            clu.get_internal_indices(cap_x, k_means.labels_, type_of_clustering='prototype_based', metric='euclidean')

        fitted_k_means_dict[n_clusters] = k_means
        df_row_dict_list.append(
            {
                'n_clusters': n_clusters,
                'inertia': k_means.inertia_,
                'calinski_harabasz_score': k_means_internal_indices_dict['calinski_harabasz_score'],
                'davies_bouldin_score': k_means_internal_indices_dict['davies_bouldin_score'],
                'silhouette_score': k_means_internal_indices_dict['silhouette_score']
            }
        )
    results_df = pd.DataFrame(df_row_dict_list)

    # plot inertia vs n_clusters
    print('\n')
    sns.lineplot(data=results_df, x='n_clusters', y='inertia')
    plt.title('inertia vs n_clusters\nused to determine number of clusters by finding elbow')
    plt.grid()
    plt.show()

    # determine elbow location and run slope change test
    n_clusters = \
        clu.locate_elbow_in_inertia_vs_n_clusters_curve(results_df, find_right_sensitivity=find_right_sensitivity,
                                                        sensitivity=sensitivity)
    slope_change_at_elbow_test_results, elbow_slope_diff = \
        clu.test_slope_change_at_elbow(results_df, n_clusters, min_elbow_slope_diff=min_elbow_slope_diff)

    # enhanced k-means clustering with algorithms to further validate findings - sometimes a poor clustering solution
    # gets turned into a good clustering solution at the cost of complexity
    if enhanced_k_means:
        slope_change_at_elbow_test_results, n_clusters = \
            enhanced_k_means_clustering_1(slope_change_at_elbow_test_results, results_df, n_clusters)

    return {
        'cap_x': cap_x,
        'y': y,
        'n_clusters': n_clusters,
        'slope_change_at_elbow_test_results': slope_change_at_elbow_test_results,
        'elbow_slope_diff': elbow_slope_diff,
        'min_elbow_slope_diff': min_elbow_slope_diff,
        'fitted_k_means_dict': fitted_k_means_dict
    }


def print_1():
    # getting this out of the logic flow, so I can follow the logic
    print(f'\nk-means might be a good clustering tool for this data - check to see if internal indices agree with '
          f'n_clusters from elbow method')


def print_2():
    # getting this out of the logic flow, so I can follow the logic
    print(f'\nall internal indices agree with the elbow method - k-means is a good clustering method for this '
          f'data')


def print_3():
    # getting this out of the logic flow, so I can follow the logic
    print(f'\nall of the internal indices k values do not agree with the elbow method k - '
          f'k-means may not be a good clustering method for this data')
    print(f'\ndo all of the internal indices and max elbow_slope_diff give the same number of clusters?')


def print_4():
    # getting this out of the logic flow, so I can follow the logic
    print(f'\nk-means might be a good clustering tool for this data - all internal indices agree and so does '
          f'elbow_slope_diff')


def print_5():
    # getting this out of the logic flow, so I can follow the logic
    print(f'\nthe internal indices do not agree with each other - this indicate that k-means is not a good '
          f'clustering tool for this data set\n')


def enhanced_k_means_clustering_1(slope_change_at_elbow_test_results, results_df, n_clusters):

    if slope_change_at_elbow_test_results == 'pass':

        print_1()

        # this block of code checks if elbow method got it right by checking if the internal indices agree with the
        # elbow method n_clusters
        ii_n_clusters_dict, ii_n_clusters_list = get_n_clusters_indicated_by_internal_indices(results_df)
        print('\n', ii_n_clusters_dict, sep='')
        if all(n_clusters == ii_n_clusters for ii_n_clusters in ii_n_clusters_list):

            # internal indices and elbow method agree on n_clusters - k-means is good method for this data set
            print_2()
            plot_internal_indices(results_df)

            return slope_change_at_elbow_test_results, n_clusters

    # the following code assesses the k-means solution further - we don't want to throw away what might be a good
    # solution
    print_3()

    # leave n_clusters from elbow method out - see if max slope change and the internal indices all agree on n_clusters
    results_df = clu.get_slope_change_all_interior_n_clusters(results_df)
    ii_n_clusters = use_internal_indices_to_get_n_clusters(results_df)
    if ii_n_clusters is not None:
        # max slope change and the internal indices all agree on n_clusters
        print_4()
        plot_internal_indices(results_df)
        slope_change_at_elbow_test_results = 'pass'
        n_clusters = ii_n_clusters
    else:
        # max slope change and the internal indices do not agree on n_clusters
        print_5()
        slope_change_at_elbow_test_results = 'fail'

    return slope_change_at_elbow_test_results, n_clusters


def plot_internal_indices(results_df):

    for internal_index in ['calinski_harabasz_score', 'davies_bouldin_score', 'silhouette_score', 'elbow_slope_diff']:
        if internal_index in results_df.columns:
            print('\n')
            sns.lineplot(data=results_df, x='n_clusters', y=internal_index)
            plt.title(f'{internal_index} vs n_clusters')
            plt.grid()
            plt.show()


def get_n_clusters_indicated_by_internal_indices(results_df):

    ii_n_clusters_list = []
    ii_n_clusters_dict = {}
    for attr in results_df.columns:

        if attr in ['calinski_harabasz_score', 'davies_bouldin_score', 'silhouette_score', 'elbow_slope_diff']:

            if attr in ['calinski_harabasz_score', 'silhouette_score', 'elbow_slope_diff']:
                idx = results_df[attr].idxmax()
            else:
                idx = results_df[attr].idxmin()

            n_clusters = results_df.loc[idx, 'n_clusters']
            ii_n_clusters_list.append(n_clusters)
            ii_n_clusters_dict[attr] = n_clusters

    return ii_n_clusters_dict, ii_n_clusters_list


def use_internal_indices_to_get_n_clusters(results_df):

    ii_n_clusters_dict, ii_n_clusters_list = get_n_clusters_indicated_by_internal_indices(results_df)
    print('\n', ii_n_clusters_dict, sep='')

    n_clusters = None
    if np.unique(np.array(ii_n_clusters_list)).shape[0] == 1:  # all internal indices agree and so does elbow_slope_diff
        n_clusters = ii_n_clusters_list[0]

    return n_clusters


if __name__ == "__main__":
    pass
