import pandas as pd
import sys
import numpy as np
from sklearn.metrics.cluster import (contingency_matrix, pair_confusion_matrix, rand_score, adjusted_rand_score,
                                     adjusted_mutual_info_score, mutual_info_score, normalized_mutual_info_score,
                                     homogeneity_completeness_v_measure, fowlkes_mallows_score)
import seaborn as sns
import matplotlib.pyplot as plt
import itertools


def analyze_labels(labels, type_of_labels=None):
    """

    :param labels:
    :param type_of_labels: 'clusters' or 'class'
    :return:
    """

    if type_of_labels == 'clusters':  # labels_pred
        things = 'clusters'
    elif type_of_labels == 'class':  # labels_true
        things = 'true classes'
    else:
        sys.exit(f'{type_of_labels} are not recognized')

    labels = pd.Series(labels)

    print(f'\n', '*' * 80, sep='')
    print(f'number of {things}: {labels.nunique()}')
    print(f'value counts of {things}:\n{labels.value_counts()}')


def get_clustering_contingency_matrix(labels_pred, labels_true, print_out=False):

    contingency_matrix_ = contingency_matrix(labels_true, labels_pred)

    if print_out:
        print('\n', '*' * 50, sep='')
        print(f'clustering_contingency_matrix')
        print(f'\nlabels_true has {len(set(labels_true))} classes - these form the rows of the contingency matrix')
        print(f'labels_pred has {len(set(labels_pred))} clusters - these form the columns of the contingency matrix')
        print(f'\ncontingency_matrix:\n{contingency_matrix_}')
        # cluster_class_dist_matrix = contingency_matrix_/contingency_matrix_.sum(axis=0)
        # print(f'\ncluster_class_dist_matrix:\n{cluster_class_dist_matrix}')

    return {
        'contingency_matrix': contingency_matrix_
    }


def get_the_f_measure_of_cluster_i_wrt_class_j(cluster_i, class_j, labels_pred, labels_true, print_out=False):
    """
    A combination of both precision and recall that measures the extent to which a cluster contains only objects of a
    particular class and all objects of that class. The f-measure of cluster i with respect to class j is f_measure_ij.

    :param cluster_i: cluster number - integer as string
    :param class_j: class number - integer as string
    :param labels_pred:
    :param labels_true:
    :param print_out:
    :return:
    """

    # get quantities for computation
    results_dict = get_precision_of_cluster_i_wrt_class_j(cluster_i, class_j, labels_pred, labels_true,
                                                          print_out=print_out)
    precision_ij = results_dict['precision_ij']

    results_dict = get_recall_of_cluster_i_wrt_class_j(cluster_i, class_j, labels_pred, labels_true,
                                                       print_out=print_out)
    recall_ij = results_dict['recall_ij']

    if (precision_ij + recall_ij) == 0:
        f_measure_ij = np.nan
    else:
        f_measure_ij = (2 * precision_ij * recall_ij) / (precision_ij + recall_ij)

    return {
        'cluster_i': cluster_i,
        'class_j': class_j,
        'f_measure_ij': f_measure_ij
    }


def get_precision_of_cluster_i_wrt_class_j(cluster_i, class_j, labels_pred, labels_true, print_out=False):
    """
    The fraction of a cluster that consists of objects of a specified class. The precision of cluster i with
    respect to class j is precision_ij.

    :param cluster_i: cluster number - integer as string
    :param class_j: class number - integer as string
    :param labels_pred:
    :param labels_true:
    :param print_out:

    :return: precision_ij
    """

    # get required base quantities for the entropy calculation
    results_dict = get_cluster_class_dist_dict(labels_pred, labels_true, print_out=print_out)
    cluster_class_dist_dict = results_dict['cluster_class_dist_dict']

    precision_ij = None
    cluster_keys = list(cluster_class_dist_dict.keys())
    for cluster_key in cluster_keys:
        if cluster_i in cluster_key:
            class_keys = list(cluster_class_dist_dict[cluster_key].keys())
            for class_key in class_keys:
                if class_j in class_key:
                    precision_ij = cluster_class_dist_dict[cluster_key][class_key]
                    break
    return {
        'cluster_i': cluster_i,
        'class_j': class_j,
        'precision_ij': precision_ij
    }


def get_recall_of_cluster_i_wrt_class_j(cluster_i, class_j, labels_pred, labels_true, print_out=False):
    """
    The extent to which a cluster contains all objects of a specified class. The recall of cluster i with respect to
    class j is recall_ij.

    :param cluster_i: cluster number - integer as string
    :param class_j: class number - integer as string
    :param labels_pred:
    :param labels_true:
    :param print_out:
    :return:
    """

    # get required base quantities for calculation
    results_dict = get_cluster_class_dist_dict(labels_pred, labels_true, print_out=print_out)
    clustering_data_object_counts_dict = results_dict['clustering_data_object_counts_dict']
    class_j_count_in_cluster_i_aka_m_ij_dict = results_dict['class_j_count_in_cluster_i_aka_m_ij_dict']

    results_dict = get_class_data_object_counts_dict(labels_true)
    class_data_object_counts_dict = results_dict['class_data_object_counts_dict']

    if print_out:
        print(f'\nclustering_data_object_counts_dict:\n{clustering_data_object_counts_dict}')
        print(f'\nclass_j_count_in_cluster_i_aka_m_ij_dict:\n{class_j_count_in_cluster_i_aka_m_ij_dict}')
        print(f'\nclass_data_object_counts_dict:\n{class_data_object_counts_dict}')

    recall_ij = None
    cluster_keys = list(clustering_data_object_counts_dict.keys())
    for cluster_key in cluster_keys:
        if cluster_i in cluster_key:
            class_keys = list(class_j_count_in_cluster_i_aka_m_ij_dict[cluster_key].keys())
            for class_key in class_keys:
                if class_j in class_key:
                    m_j = class_data_object_counts_dict[class_key]
                    m_ij = class_j_count_in_cluster_i_aka_m_ij_dict[cluster_key][class_key]
                    recall_ij = m_ij/m_j
                    break
    return {
        'cluster_i': cluster_i,
        'class_j': class_j,
        'recall_ij': recall_ij
    }


def get_clustering_purity_alt_algo(labels_pred, labels_true):

    # https://stackoverflow.com/questions/34047540/python-clustering-purity-metric

    return_dict = get_clustering_contingency_matrix(labels_pred, labels_true)
    contingency_matrix_ = return_dict['contingency_matrix']

    purity = np.sum(np.amax(contingency_matrix_, axis=0))/np.sum(contingency_matrix_)

    return {
        'purity': purity
    }


def get_clustering_purity(labels_pred, labels_true, print_out=False):

    # get required base quantities for the entropy calculation
    results_dict = get_cluster_class_dist_dict(labels_pred, labels_true, print_out=print_out)
    cluster_class_dist_dict = results_dict['cluster_class_dist_dict']
    clustering_data_object_counts_dict = results_dict['clustering_data_object_counts_dict']

    # get the total number of data objects in the data set
    num_data_objects_in_data_set = sum(clustering_data_object_counts_dict.values())

    # compute purity
    cluster_purity_dict = {}
    clustering_purity = 0
    for cluster_id, cluster_class_dist in cluster_class_dist_dict.items():
        cluster_purity = max(cluster_class_dist.values())
        cluster_purity_dict[cluster_id] = cluster_purity
        clustering_purity += \
            ((clustering_data_object_counts_dict[cluster_id]/num_data_objects_in_data_set) * cluster_purity)

    if print_out:
        print(f'\ncluster_purity_dict:\n{cluster_purity_dict}')
        print(f'\nclustering_purity:\n{cluster_purity_dict}')

    return {
        'clustering_data_object_counts_dict': clustering_data_object_counts_dict,
        'cluster_class_dist_dict': cluster_class_dist_dict,
        'cluster_purity_dict': cluster_purity_dict,
        'clustering_purity': clustering_purity
    }


def form_a_data_frame_with_labels(labels_pred, labels_true, print_out=False):

    # bring the pred and true labels together
    df = pd.DataFrame(
        {
            'pred': labels_pred,
            'true': labels_true
        }
    )

    if print_out:
        max_rows = 10
        if df.shape[0] > max_rows:
            print(f'labels_data_frame - df.head({max_rows}):\n\n{df.head(max_rows)}\n')
        else:
            print(f'labels_data_frame:\n\n{df}\n')

    return {
        'labels_data_frame': df
    }


def get_class_data_object_counts_dict(labels_true):

    value_counts = pd.Series(labels_true).value_counts()
    class_id_key_list = []
    for class_id in value_counts.index:
        class_id_key_list.append(get_class_id_key(class_id))
    class_data_object_counts_dict = dict(zip(class_id_key_list, value_counts.values))

    return {
        'class_data_object_counts_dict': class_data_object_counts_dict
    }


def get_class_id_key(class_id):
    class_id_key = 'class_' + str(class_id)
    return class_id_key


def get_cluster_id_key(cluster_id):
    class_id_key = 'cluster_' + str(cluster_id)
    return class_id_key


def get_cluster_class_dist_dict(labels_pred, labels_true, print_out=False):

    # bring the pred and true labels together
    results_dict = form_a_data_frame_with_labels(labels_pred, labels_true, print_out=print_out)
    df = results_dict['labels_data_frame']

    cluster_class_dist_dict = {}
    clustering_data_object_counts_dict = {}
    class_j_count_in_cluster_i_aka_m_ij_dict = {}
    for cluster_id in df.pred.unique():  # calculate entropy for each cluster
        cluster_id_key = get_cluster_id_key(cluster_id)

        # get number of objects in cluster
        cluster_i_df = df.loc[df.pred == cluster_id]
        m_i = cluster_i_df.shape[0]
        clustering_data_object_counts_dict[cluster_id_key] = m_i

        cluster_class_dist_dict[cluster_id_key] = {}
        class_j_count_in_cluster_i_aka_m_ij_dict[cluster_id_key] = {}
        for class_id in df.true.unique():
            class_id_key = get_class_id_key(class_id)

            # how many objects of class_id are in cluster_id
            m_ij = cluster_i_df.loc[cluster_i_df.true == class_id].shape[0]
            class_j_count_in_cluster_i_aka_m_ij_dict[cluster_id_key][class_id_key] = m_ij

            # calculate class_id probability of being in luster_id - this forms the clusters class distribution
            p_ij = m_ij / m_i
            cluster_class_dist_dict[cluster_id_key][class_id_key] = p_ij

    if print_out:
        print(f'\nclustering_data_object_counts_dict:\n{clustering_data_object_counts_dict}')
        print(f'\ncluster_class_dist_dict:\n{cluster_class_dist_dict}')

    return {
        'clustering_data_object_counts_dict': clustering_data_object_counts_dict,
        'cluster_class_dist_dict': cluster_class_dist_dict,
        'class_j_count_in_cluster_i_aka_m_ij_dict': class_j_count_in_cluster_i_aka_m_ij_dict
    }


def plot_cluster_class_dist(labels_pred, labels_true, cluster_id_list=None, print_out=False):
    """

    :param labels_pred: cluster labels
    :param labels_true: class labels
    :param cluster_id_list: list of integers representing cluster identifications
    :param print_out:
    :return:
    """

    # load labels into results dict
    results_dict = form_a_data_frame_with_labels(labels_pred, labels_true, print_out=print_out)
    df = results_dict['labels_data_frame']

    # plot cluster class distribution for which clusters?
    if cluster_id_list is None:
        cluster_id_list = list(df.pred.unique())

    for cluster_id in sorted(cluster_id_list):  # cycle through clusters

        # make a class id histogram for cluster
        df_row_dict_list = []
        cluster_df = df.loc[df.pred == cluster_id].true.to_frame()
        for class_id in df.true.unique():
            num_class_id = cluster_df.loc[cluster_df.true == class_id].shape[0]
            df_row_dict_list.append(
                {
                    'class_id': str(class_id),
                    'frequency': num_class_id
                }
            )
        results_df = pd.DataFrame(df_row_dict_list).sort_values('class_id')

        # plot a class id histogram for cluster
        sns.histplot(
            data=results_df,
            x='class_id',
            stat='probability',
            weights='frequency',
        )
        plt.title(f'class distribution for cluster {cluster_id}')
        plt.grid()
        plt.show()


def get_clustering_entropy(labels_pred, labels_true, print_out=False):

    # https://stackoverflow.com/questions/35709562/how-to-calculate-clustering-entropy-a-working-example-or-software-
    # code

    # get required base quantities for the entropy calculation
    return_dict = get_cluster_class_dist_dict(labels_pred, labels_true, print_out=print_out)
    cluster_class_dist_dict = return_dict['cluster_class_dist_dict']
    clustering_data_object_counts_dict = return_dict['clustering_data_object_counts_dict']

    # get total number of data objects in the data set
    num_data_objects = sum(clustering_data_object_counts_dict.values())

    # compute entropy
    cluster_entropy_dict = {}
    entropy = 0
    for cluster_id in cluster_class_dist_dict.keys():  # calculate entropy for each cluster

        num_data_objects_in_cluster = clustering_data_object_counts_dict[cluster_id]
        cluster_entropy_dict[cluster_id] = 0
        for class_id in cluster_class_dist_dict[cluster_id].keys():

            p_ij = cluster_class_dist_dict[cluster_id][class_id]
            if p_ij != 0:  # do not include terms with p_ij = 0

                # calculate this class_id / cluster_id contribution to entropy
                e_i = -1 * p_ij * np.log2(p_ij)

                cluster_entropy_dict[cluster_id] += e_i
                entropy += (num_data_objects_in_cluster/num_data_objects) * e_i

    if print_out:
        print(f'\ncluster_class_dist_dict:\n{cluster_class_dist_dict}')
        print(f'\ncluster_entropy_dict:\n{cluster_entropy_dict}')
        print(f'\nentropy:\n{entropy}')

    return {
        'clustering_data_object_counts_dict': clustering_data_object_counts_dict,
        'cluster_class_dist_dict': cluster_class_dist_dict,
        'cluster_entropy_dict': cluster_entropy_dict,
        'entropy': entropy
    }


def get_clustering_precision(labels_pred, labels_true, print_out=False):
    precision_dict = {}
    for cluster_i in set(labels_pred):
        for class_j in set(labels_true):
            results_dict = get_precision_of_cluster_i_wrt_class_j(
                str(cluster_i),
                str(class_j),
                labels_pred,
                labels_true,
                print_out=print_out
            )
            precision_ij = results_dict['precision_ij']
            precision_dict['precision_' + str(cluster_i) + str(class_j)] = precision_ij
            if print_out:
                print(f'precision of cluster {str(cluster_i)} with respect to class {str(class_j)} is {precision_ij}')
    return {
        'precision_dict': precision_dict
    }


def get_clustering_recall(labels_pred, labels_true, print_out=False):
    recall_dict = {}
    for cluster_i in set(labels_pred):
        for class_j in set(labels_true):
            results_dict = get_recall_of_cluster_i_wrt_class_j(
                str(cluster_i),
                str(class_j),
                labels_pred,
                labels_true,
                print_out=print_out
            )
            recall_ij = results_dict['recall_ij']
            recall_dict['recall_' + str(cluster_i) + str(class_j)] = recall_ij
            if print_out:
                print(f'recall of cluster {str(cluster_i)} with respect to class {str(class_j)} is {recall_ij}')
    return {
        'recall_dict': recall_dict
    }


def get_clustering_f_measure(labels_pred, labels_true, print_out=False):

    f_measure = 0
    f_measure_dict = {}
    for class_j in set(labels_true):
        max_f_ij = 0  # maximum is taken over all clusters i in a class j
        for cluster_i in set(labels_pred):
            results_dict = get_the_f_measure_of_cluster_i_wrt_class_j(
                str(cluster_i),
                str(class_j),
                labels_pred,
                labels_true,
                print_out=print_out
            )
            f_measure_ij = results_dict['f_measure_ij']
            f_measure_dict['f_measure_' + str(cluster_i) + str(class_j)] = f_measure_ij

            if f_measure_ij > max_f_ij:
                max_f_ij = f_measure_ij

            if print_out:
                print(f'recall of cluster {str(cluster_i)} with respect to class {str(class_j)} is {f_measure_ij}')

        m_j = labels_true.count(class_j)
        f_measure += (m_j/len(labels_pred)) * max_f_ij

    return {
        'f_measure_dict': f_measure_dict,
        'f_measure': f_measure
    }


def get_similarity_matrix(labels):

    similarity_matrix = np.zeros((len(labels), len(labels)))

    for i in range(len(labels)):
        for j in range(len(labels)):
            if labels[i] == labels[j]:
                similarity_matrix[i, j] = 1

    return similarity_matrix


def get_f_00(sim_mat_1, sim_mat_2):
    """

    :param sim_mat_1: ideal_cluster_similarity_matrix
    :param sim_mat_2: class__similarity_matrix
    :return: f_00 - number of pairs of objects having a different class and a different cluster
    """

    sim_mat_1_mask = sim_mat_1 == 0
    sim_mat_2_mask = sim_mat_2 == 0
    f_00 = (sim_mat_1_mask & sim_mat_2_mask).sum()/2

    return f_00


def get_f_01(sim_mat_1, sim_mat_2):
    """

    :param sim_mat_1: ideal_cluster_similarity_matrix
    :param sim_mat_2: class__similarity_matrix
    :return: f_01 - number of pairs of objects having a different class and the same cluster
    """

    sim_mat_1_mask = sim_mat_1 == 1
    sim_mat_2_mask = sim_mat_2 == 0
    f_01 = (sim_mat_1_mask & sim_mat_2_mask).sum()/2

    return f_01


def get_f_10(sim_mat_1, sim_mat_2):
    """

    :param sim_mat_1: ideal_cluster_similarity_matrix
    :param sim_mat_2: class__similarity_matrix
    :return: f_10 - number of pairs of objects having the same class and a different cluster
    """

    sim_mat_1_mask = sim_mat_1 == 0
    sim_mat_2_mask = sim_mat_2 == 1
    f_10 = (sim_mat_1_mask & sim_mat_2_mask).sum()/2

    return f_10


def get_f_11(sim_mat_1, sim_mat_2):
    """

    :param sim_mat_1: ideal_cluster_similarity_matrix
    :param sim_mat_2: class__similarity_matrix
    :return: f_11 - number of pairs of objects having the same class and the same cluster
    """

    sim_mat_1_mask = sim_mat_1 == 1
    sim_mat_2_mask = sim_mat_2 == 1
    f_11 = (sim_mat_1_mask & sim_mat_2_mask).sum()
    f_11 = (f_11 - sim_mat_1.shape[0])/2  # subtract off the diagonal contribution

    return f_11


def get_elements_of_contingency_table(labels_pred, labels_true):

    ideal_cluster_similarity_matrix = get_similarity_matrix(labels_pred)
    class_similarity_matrix = get_similarity_matrix(labels_true)

    f_00 = get_f_00(ideal_cluster_similarity_matrix, class_similarity_matrix)
    f_01 = get_f_01(ideal_cluster_similarity_matrix, class_similarity_matrix)
    f_10 = get_f_10(ideal_cluster_similarity_matrix, class_similarity_matrix)
    f_11 = get_f_11(ideal_cluster_similarity_matrix, class_similarity_matrix)

    return f_00, f_01, f_10, f_11


def print_out_kumar_contingency_table(f_00, f_01, f_10, f_11):

    print('\n', '*' * 50, sep='')
    print(f'kumar contingency table (pair_confusion_matrix in different format)\nsee kumar table 7.12')
    print(f'\nkumar contingency table:\n{np.array([(f_11, f_10), (f_01, f_00)])}')


def get_rand_statistic(labels_pred, labels_true):

    f_00, f_01, f_10, f_11 = get_elements_of_contingency_table(labels_pred, labels_true)
    rand_statistic = (f_00 + f_11)/(f_00 + f_01 + f_10 + f_11)

    # print_out_kumar_contingency_table(f_00, f_01, f_10, f_11)

    return {
        'rand_statistic': rand_statistic
    }


def get_jaccard_coefficient(labels_pred, labels_true):

    f_00, f_01, f_10, f_11 = get_elements_of_contingency_table(labels_pred, labels_true)
    jaccard_coefficient = f_11/(f_01 + f_10 + f_11)

    # print_out_kumar_contingency_table(f_00, f_01, f_10, f_11)

    return {
        'jaccard_coefficient': jaccard_coefficient
    }


def run_all_kumar_external_indices_helper(labels_pred, labels_true, print_out=False):

    df_row_dict = {}

    # get entropy
    return_dict = get_clustering_entropy(labels_pred, labels_true, print_out=print_out)
    df_row_dict['entropy'] = return_dict['entropy']

    # get purity
    return_dict = get_clustering_purity(labels_pred, labels_true, print_out=print_out)
    df_row_dict['purity'] = return_dict['clustering_purity']

    # get precision
    return_dict = get_clustering_precision(labels_pred, labels_true, print_out=print_out)
    df_row_dict['precision_dict'] = return_dict['precision_dict']

    # get recall
    return_dict = get_clustering_recall(labels_pred, labels_true, print_out=print_out)
    df_row_dict['recall_dict'] = return_dict['recall_dict']

    # get f-measure
    return_dict = get_clustering_f_measure(labels_pred, labels_true, print_out=print_out)
    df_row_dict['f_measure_dict'] = return_dict['f_measure_dict']

    # get rand statistic
    return_dict = get_rand_statistic(labels_pred, labels_true)
    df_row_dict['rand_statistic'] = return_dict['rand_statistic']

    # get jaccard coefficient
    return_dict = get_jaccard_coefficient(labels_pred, labels_true)
    df_row_dict['jaccard_coefficient'] = return_dict['jaccard_coefficient']

    return {
        'df_row_dict': df_row_dict
    }


def plot_run_all_kumar_results(results_df):

    # plot entropy and purity
    temp_results_df = results_df[['data_set_name', 'entropy', 'purity']]
    print(f'\n{temp_results_df}\n')
    temp_results_df = (pd.melt(temp_results_df, id_vars='data_set_name', value_vars=['entropy', 'purity']).
                       rename(columns={'variable': 'external_index'}))
    sns.catplot(data=temp_results_df, x='data_set_name', y='value', hue='external_index', kind='bar')
    plt.xticks(rotation=90)
    plt.grid()
    plt.show()

    # plot rand statistic and jaccard coefficient
    temp_results_df = results_df[['data_set_name', 'rand_statistic', 'jaccard_coefficient']]
    print(f'\n{temp_results_df}\n')
    temp_results_df = (pd.melt(temp_results_df, id_vars='data_set_name', value_vars=['rand_statistic',
                                                                                     'jaccard_coefficient']).
                       rename(columns={'variable': 'external_index'}))
    sns.catplot(data=temp_results_df, x='data_set_name', y='value', hue='external_index', kind='bar')
    plt.xticks(rotation=90)
    plt.grid()
    plt.show()

    # plot precision
    print(f'\n', '*' * 80, sep='')
    print(f'precision')
    precision_dicts = results_df.loc[:, 'precision_dict']
    data_set_name = results_df.loc[:, 'data_set_name']
    for data_set_name, precision_dict in zip(data_set_name, precision_dicts):
        print(f'\n', '*' * 60, sep='')
        print(f'data_set_name: {data_set_name}')
        print(f'\n{precision_dict}\n')
        temp_df = (pd.DataFrame(precision_dict, index=[0]).T.reset_index().rename(columns={'index': 'precision',
                                                                                           0: 'value'}))
        sns.catplot(data=temp_df, x='precision', y='value', kind='bar', color='b')
        plt.title(f'{data_set_name}')
        plt.xticks(rotation=90)
        plt.xlabel('precision_ij -> precision of cluster i wrt class j')
        plt.ylabel('precision')
        plt.grid()
        plt.ylim([0, 1])
        plt.show()

    # plot recall
    print(f'\n', '*' * 80, sep='')
    print(f'recall')
    recall_dicts = results_df.loc[:, 'recall_dict']
    data_set_name = results_df.loc[:, 'data_set_name']
    for data_set_name, recall_dict in zip(data_set_name, recall_dicts):
        print(f'\n', '*' * 60, sep='')
        print(f'data_set_name: {data_set_name}')
        print(f'\n{recall_dict}\n')
        temp_df = (pd.DataFrame(recall_dict, index=[0]).T.reset_index().rename(columns={'index': 'recall',
                                                                                        0: 'value'}))
        sns.catplot(data=temp_df, x='recall', y='value', kind='bar', color='b')
        plt.title(f'{data_set_name}')
        plt.xticks(rotation=90)
        plt.xlabel('recall_ij -> recall of cluster i wrt class j')
        plt.ylabel('recall')
        plt.grid()
        plt.ylim([0, 1])
        plt.show()

    # plot f-measure
    print(f'\n', '*' * 80, sep='')
    print(f'f_measure')
    f_measure_dicts = results_df.loc[:, 'f_measure_dict']
    data_set_name = results_df.loc[:, 'data_set_name']
    for data_set_name, f_measure_dict in zip(data_set_name, f_measure_dicts):
        print(f'\n', '*' * 60, sep='')
        print(f'data_set_name: {data_set_name}')
        print(f'\n{f_measure_dict}\n')
        temp_df = (pd.DataFrame(f_measure_dict, index=[0]).T.reset_index().rename(columns={'index': 'f_measure',
                                                                                           0: 'value'}))
        sns.catplot(data=temp_df, x='f_measure', y='value', kind='bar', color='b')
        plt.title(f'{data_set_name}')
        plt.xticks(rotation=90)
        plt.xlabel('f_measure_ij -> f_measure of cluster i wrt class j')
        plt.ylabel('f_measure')
        plt.grid()
        plt.ylim([0, 1])
        plt.show()


def run_all_kumar_external_indices_on_data_set_dict(data_set_dict, print_out=False):

    df_row_dict_list = []
    for data_set_name, data_set in data_set_dict.items():

        print(f'\n', '*' * 80, sep='')
        print(f'data_set_name: {data_set_name}')

        labels_true = data_set['labels_true']
        labels_pred = data_set['labels_pred']

        if print_out:
            print(f'\nlabels_true has {len(set(labels_true))} classes')
            print(f'labels_true:\n{labels_true}')
            print(f'\nlabels_pred has {len(set(labels_pred))} clusters')
            print(f'labels_pred:\n{labels_pred}')

        results_dict = run_all_kumar_external_indices_helper(labels_pred, labels_true, print_out=print_out)
        df_row_dict = results_dict['df_row_dict']

        df_row_dict['data_set_name'] = data_set_name
        df_row_dict_list.append(df_row_dict)

    results_df = pd.DataFrame(df_row_dict_list)
    results_df = results_df[['data_set_name'] + [attr for attr in results_df.columns if attr != 'data_set_name']]

    plot_run_all_kumar_results(results_df)

    return {
        'data_set_dict': data_set_dict,
        'results_df': results_df
    }


def get_pair_confusion_matrix(labels_true, labels_pred,  print_out=False):

    pair_confusion_matrix_ = pair_confusion_matrix(labels_true, labels_pred)
    if print_out:
        print('\n', '*' * 50, sep='')
        print(f'pair_confusion_matrix')
        print(f'\npair_confusion_matrix:\n{pair_confusion_matrix_}')

    return {
        'pair_confusion_matrix': pair_confusion_matrix_
    }


def get_rand_index(labels_pred, labels_true, print_out=False):

    rand_score_ = rand_score(labels_true, labels_pred)

    if print_out:
        print('\n', '*' * 50, sep='')
        print(f'rand_index')
        print(f'\nrand_score: {rand_score_}')

    return {
        'rand_score': rand_score_
    }


def get_adjusted_rand_index(labels_pred, labels_true, print_out=False):

    adjusted_rand_score_ = adjusted_rand_score(labels_true, labels_pred)

    if print_out:
        print('\n', '*' * 50, sep='')
        print(f'adjusted_rand_index')
        print(f'\nadjusted_rand_score: {adjusted_rand_score_}')

    return {
        'adjusted_rand_score': adjusted_rand_score_
    }


def get_mutual_info_based_score(labels_pred, labels_true, print_out=False):

    mutual_info_score_ = mutual_info_score(labels_true, labels_pred, contingency=None)
    adjusted_mutual_info_score_ = adjusted_mutual_info_score(labels_true, labels_pred, average_method='arithmetic')
    normalized_mutual_info_score_ = normalized_mutual_info_score(labels_true, labels_pred, average_method='arithmetic')

    if print_out:
        print('\n', '*' * 50, sep='')
        print(f'mutual_info_score')
        print(f'\nmutual_info_score: {mutual_info_score_}')
        print('\n', '*' * 50, sep='')
        print(f'normalized_mutual_info_score')
        print(f'\nnormalized_mutual_info_score: {normalized_mutual_info_score_}')
        print('\n', '*' * 50, sep='')
        print(f'adjusted_mutual_info_score')
        print(f'\nadjusted_mutual_info_score: {adjusted_mutual_info_score_}')

    return {
        'mutual_info_score': mutual_info_score_,
        'normalized_mutual_info_score': normalized_mutual_info_score_,
        'adjusted_mutual_info_score': adjusted_mutual_info_score_
    }


def get_homogeneity_completeness_score_and_v_measure(labels_pred, labels_true, beta=1.0, print_out=False):

    homogeneity_score_, completeness_score_, v_measure_ = (
        homogeneity_completeness_v_measure(labels_true, labels_pred, beta=beta))

    if print_out:
        print('\n', '*' * 50, sep='')
        print(f'homogeneity_score')
        print(f'\nhomogeneity_score: {homogeneity_score_}')
        print('\n', '*' * 50, sep='')
        print(f'completeness_score')
        print(f'\ncompleteness_score: {completeness_score_}')
        print('\n', '*' * 50, sep='')
        print(f'v_measure')
        print(f'\nv_measure: {v_measure_}')

    return {
        'homogeneity_score': homogeneity_score_,
        'completeness_score': completeness_score_,
        'v_measure': v_measure_
    }


def get_fowlkes_mallows_score(labels_pred, labels_true, print_out=False):

    fowlkes_mallows_score_ = fowlkes_mallows_score(labels_true, labels_pred, sparse=False)

    if print_out:
        print('\n', '*' * 50, sep='')
        print(f'fowlkes_mallows_score')
        print(f'\nfowlkes_mallows_score: {fowlkes_mallows_score_}')

    return {
        'fowlkes_mallows_score': fowlkes_mallows_score_
    }


def run_all_sklearn_external_indices_helper(labels_pred, labels_true, print_out=False):

    df_row_dict = {}

    # get contingency matrix
    return_dict = get_clustering_contingency_matrix(labels_pred, labels_true, print_out=print_out)
    df_row_dict['contingency_matrix'] = return_dict['contingency_matrix']

    # get pair_confusion_matrix - comparing partitions
    return_dict = get_pair_confusion_matrix(labels_true, labels_pred,  print_out=print_out)
    pair_confusion_matrix_ = return_dict['pair_confusion_matrix']
    df_row_dict['pair_confusion_matrix'] = pair_confusion_matrix_

    # rand index
    return_dict = get_rand_index(labels_pred, labels_true, print_out=print_out)
    df_row_dict['rand_score'] = return_dict['rand_score']

    # get adjusted rand index
    return_dict = get_adjusted_rand_index(labels_pred, labels_true, print_out=print_out)
    df_row_dict['adjusted_rand_score'] = return_dict['adjusted_rand_score']

    # get mutual information based scores
    return_dict = get_mutual_info_based_score(labels_pred, labels_true, print_out=print_out)
    df_row_dict['mutual_info_score'] = return_dict['mutual_info_score']
    df_row_dict['normalized_mutual_info_score'] = return_dict['normalized_mutual_info_score']
    df_row_dict['adjusted_mutual_info_score'] = return_dict['adjusted_mutual_info_score']

    # get homogeneity score
    return_dict = get_homogeneity_completeness_score_and_v_measure(labels_pred, labels_true, beta=1.0,
                                                                   print_out=print_out)
    df_row_dict['homogeneity_score'] = return_dict['homogeneity_score']
    df_row_dict['completeness_score'] = return_dict['completeness_score']
    df_row_dict['v_measure'] = return_dict['v_measure']

    # get Fowlkes-Mallows score
    return_dict = get_fowlkes_mallows_score(labels_pred, labels_true, print_out=print_out)
    df_row_dict['fowlkes_mallows_score'] = return_dict['fowlkes_mallows_score']

    return {
        'df_row_dict': df_row_dict
    }


def run_all_sklearn_external_indices_on_data_set_dict(data_set_dict, print_out=False):
    df_row_dict_list = []
    for data_set_name, data_set in data_set_dict.items():

        print(f'\n', '*' * 80, sep='')
        print(f'data_set_name: {data_set_name}')

        labels_true = data_set['labels_true']
        labels_pred = data_set['labels_pred']

        if print_out:
            print(f'\nlabels_true has {len(set(labels_true))} classes')
            print(f'labels_true:\n{labels_true}')
            print(f'\nlabels_pred has {len(set(labels_pred))} clusters')
            print(f'labels_pred:\n{labels_pred}')

        results_dict = run_all_sklearn_external_indices_helper(labels_pred, labels_true, print_out=print_out)
        df_row_dict = results_dict['df_row_dict']

        df_row_dict['data_set_name'] = data_set_name
        df_row_dict_list.append(df_row_dict)

    results_df = pd.DataFrame(df_row_dict_list)
    results_df = results_df[['data_set_name'] + [attr for attr in results_df.columns if attr != 'data_set_name']]

    return {
        'data_set_dict': data_set_dict,
        'results_df': results_df
    }


def run_all_external_indices_taught_in_spring_24_on_data_set_dict(data_set_dict, print_out=False):

    df_row_dict_list = []
    for data_set_name, data_set in data_set_dict.items():

        if data_set_name == 'kumar_table_7_10':
            continue

        print(f'\n', '*' * 80, sep='')
        print('*' * 80)
        print('*' * 80)
        print(f'data_set_name: {data_set_name}')

        labels_true = data_set['labels_true']
        labels_pred = data_set['labels_pred']

        if print_out:
            print(f'\nlabels_true has {len(set(labels_true))} classes')
            print(f'labels_true:\n{labels_true}')
            print(f'\nlabels_pred has {len(set(labels_pred))} clusters')
            print(f'labels_pred:\n{labels_pred}')

        results_dict = external_indices_taught_spring_24(labels_pred, labels_true, print_out=print_out)
        df_row_dict = results_dict['df_row_dict']

        df_row_dict['data_set_name'] = data_set_name
        df_row_dict_list.append(df_row_dict)

    results_df = pd.DataFrame(df_row_dict_list)
    results_df = results_df[['data_set_name'] + [attr for attr in results_df.columns if attr != 'data_set_name']]

    plot_spring_24_results(results_df)

    return {
        'data_set_dict': data_set_dict,
        'results_df': results_df
    }


def plot_spring_24_results(results_df):

    # rand score and adjusted rand score
    temp_results_df = results_df[['data_set_name', 'rand_score', 'adjusted_rand_score']]
    print(f'\n{temp_results_df}\n')
    temp_results_df = (pd.melt(temp_results_df, id_vars='data_set_name',
                               value_vars=['rand_score', 'adjusted_rand_score']).
                       rename(columns={'variable': 'external_index'}))
    sns.catplot(data=temp_results_df, x='data_set_name', y='value', hue='external_index', kind='bar')
    plt.xticks(rotation=90)
    plt.grid()
    plt.show()


def external_indices_taught_spring_24(labels_pred, labels_true, print_out=False):

    df_row_dict = {}

    # rand score
    return_dict = get_rand_index(labels_pred, labels_true, print_out=print_out)
    df_row_dict['rand_score'] = return_dict['rand_score']

    # adjusted rand score
    return_dict = get_adjusted_rand_index(labels_pred, labels_true, print_out=print_out)
    df_row_dict['adjusted_rand_score'] = return_dict['adjusted_rand_score']

    # cluster contingency matrix using best cluster labeling
    return_dict = get_best_cluster_label_permutation(labels_pred, labels_true, print_out=False)
    labels_pred = return_dict['best_perm_labels_pred']
    return_dict = get_clustering_contingency_matrix(labels_pred, labels_true, print_out=print_out)
    df_row_dict['contingency_matrix'] = return_dict['contingency_matrix']

    return {
        'df_row_dict': df_row_dict
    }


def get_best_cluster_labels_print_1(labels_true, labels_pred, perm_labels_pred, mapping):

    if labels_pred == perm_labels_pred:
        print(f'\noriginal - labels_pred are not permuted')
        print(f'labels_true: {labels_true}\nlabels_pred: {labels_pred}')
    else:
        print(f'\nlabels_pred are permuted')
        print(f'labels_true:      {labels_true}\nlabels_pred:      {labels_pred}\n'
              f'perm_labels_pred: {perm_labels_pred}')
        print(f'mapping (original -> permuted): {mapping}')


def check_out_permutation_of_cluster_labels(labels_pred, labels_true, print_out=False):

    df_row_dict_list = []
    i = 0
    for cluster_labels in itertools.permutations(set(labels_pred)):

        print('\n', '*' * 80, sep='')

        mapping = dict(zip(set(labels_pred), cluster_labels))
        perm_labels_pred = [mapping[label] for label in labels_pred]

        get_best_cluster_labels_print_1(labels_true, labels_pred, perm_labels_pred, mapping)

        return_dict = get_jaccard_coefficient(perm_labels_pred, labels_true)
        jaccard_coefficient = return_dict['jaccard_coefficient']

        rand_score_ = rand_score(labels_true, labels_pred)

        adjusted_rand_score_ = adjusted_rand_score(labels_true, labels_pred)

        return_dict = get_clustering_entropy(perm_labels_pred, labels_true, print_out=False)
        entropy = return_dict['entropy']

        return_dict = get_clustering_f_measure(labels_pred, labels_true, print_out=False)
        f_measure = return_dict['f_measure']

        return_dict = get_clustering_purity(labels_pred, labels_true, print_out=print_out)
        purity = return_dict['clustering_purity']

        return_dict = get_clustering_contingency_matrix(perm_labels_pred, labels_true, print_out=False)
        contingency_matrix_ = return_dict['contingency_matrix']

        contingency_matrix_trace = np.trace(contingency_matrix_)

        if labels_pred == perm_labels_pred:
            permutation = 'original'
        else:
            i += 1
            permutation = str(i)

        if print_out:
            print(f'\njaccard_coefficient: {jaccard_coefficient}')
            print(f'rand_score: {rand_score_}')
            print(f'adjusted_rand_score: {adjusted_rand_score_}')
            print(f'entropy: {entropy}')
            print(f'f_measure: {f_measure}')
            print(f'purity: {purity}')

        print(f'\ncontingency_matrix:\n{contingency_matrix_}')
        print(f'\ncontingency_matrix_trace:\n{contingency_matrix_trace}')

        df_row_dict_list.append(
            {
                'permutation': permutation,
                'mapping': mapping,
                'jaccard_coefficient': jaccard_coefficient,
                'rand_score': rand_score_,
                'adjusted_rand_score': adjusted_rand_score_,
                'entropy': entropy,
                'f_measure': f_measure,
                'purity': purity,
                'contingency_matrix': contingency_matrix_,
                'contingency_matrix_trace': contingency_matrix_trace
            }
        )

    results_df = pd.DataFrame(df_row_dict_list)

    return {
        'results_df': results_df
    }


def get_best_cluster_label_permutation(labels_pred, labels_true, print_out=False):

    best_permutation_mapping = None
    best_perm_labels_pred = None
    best_contingency_matrix = None

    max_contingency_matrix_trace = 0
    for cluster_labels in itertools.permutations(set(labels_pred)):

        mapping = dict(zip(set(labels_pred), cluster_labels))
        perm_labels_pred = [mapping[label] for label in labels_pred]

        return_dict = get_clustering_contingency_matrix(perm_labels_pred, labels_true, print_out=print_out)
        contingency_matrix_ = return_dict['contingency_matrix']

        contingency_matrix_trace = np.trace(contingency_matrix_)

        if print_out:
            print(f'\ncontingency_matrix:\n{contingency_matrix_}')
            print(f'\ncontingency_matrix_trace:\n{contingency_matrix_trace}')

        if contingency_matrix_trace > max_contingency_matrix_trace:
            max_contingency_matrix_trace = contingency_matrix_trace
            best_permutation_mapping = mapping
            best_perm_labels_pred = perm_labels_pred
            best_contingency_matrix = contingency_matrix_

    return {
        'best_permutation_mapping': best_permutation_mapping,
        'best_perm_labels_pred': best_perm_labels_pred,
        'best_contingency_matrix': best_contingency_matrix
    }


def remove_noise_data_objects_from_labels(labels_pred, labels_true):

    labels_pred = np.array(labels_pred).reshape(-1, 1)
    labels_true = np.array(labels_true).reshape(-1, 1)
    labels = np.concatenate((labels_pred, labels_true), axis=1)

    labels = labels[~np.any(labels == -1, axis=1), :]

    return {
        'labels_pred': labels[:, 0],
        'labels_true': labels[:, 1]
    }


if __name__ == "__main__":
    pass
