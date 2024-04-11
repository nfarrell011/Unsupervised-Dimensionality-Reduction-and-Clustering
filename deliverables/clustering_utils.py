'''
    Midterm: Clustering - Digits Dataset
    Joseph Nelson Farrell
    DS 5230 Unsupervised Machine Learning
    Northeastern University
    Steven Morin, PhD

    This file contains a library of helper functions to be used in clustering.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from kneed import KneeLocator
from sklearn.metrics.cluster import contingency_matrix, adjusted_rand_score
from scipy.stats import mode
import itertools
import time
                            
###########################################################################################################################
#########################################          get_nn_dist            #################################################
###########################################################################################################################
def get_nn_dist(embedding: np.ndarray) -> list:
    """  
        Function: get_nn_dist

        Parameters:
        __________
        
            embedding: (np.ndarray) this is an n-dimensional embedding

        Returns:
        ________

            nn_dist_list : (list) a list of nearest neighbor distances

        *** This code was adapted from code provided by Professor Steven Morin ***
    """

    # build the kdtree
    kdt = KDTree(embedding)

    # generate list of nearest neighbors
    nn_dist_list = []
    for i in range(embedding.shape[0]):
        dist, holder = kdt.query(embedding[i, :].reshape(1, -1), 2)
        nn_dist_list.append(dist[0, -1])

    return nn_dist_list

###########################################################################################################################
#####################################       generate_hopkins_statistic       ##############################################
###########################################################################################################################
def generate_hopkins_statistic(embedding: np.ndarray) -> float:
    """  
        Function: generate_hopkins_statitic

            This function will find the hopkins statistic associated with an embedding. This metric measures the propensity 
                of the data to cluster. 

        Parameters:
        __________
        
            embedding: (np.ndarray) - this is an n-dimensional embedding

        Returns:
        ________

            hopkins_stat : (float) - the Hopkins statistic

        *** This code was adapted from code provided by Professor Steven Morin ***
    """
    # get max and min
    min = embedding.min(axis = 0)
    max = embedding.max(axis = 0)

    # set seed
    np.random.seed(42)

    # create random data
    rand_x = np.random.uniform(low = min[0], high = max[0], size = (embedding.shape[0], 1))
    rand_y = np.random.uniform(low = min[1], high = max[1], size = (embedding.shape[0], 1))
    randomly_distributed_data = np.concatenate((rand_x, rand_y), axis = 1)

    # get distances lists for embedding and random data
    embedding_nn_dist_list = get_nn_dist(embedding)
    randomly_distributed_data_nn_dist_list = get_nn_dist(randomly_distributed_data)

    # compute hokins stat
    hopkins_stat = sum(embedding_nn_dist_list) / (sum(randomly_distributed_data_nn_dist_list) + sum(embedding_nn_dist_list))
    hopkins_stat = round(hopkins_stat, 3)

    return hopkins_stat

###########################################################################################################################
#####################################              find_elbow                ##############################################
###########################################################################################################################
def find_elbow(k_vs_inertia_df: pd.DataFrame) -> int:
    """
        Function: find_elbow

            This function will attempt to find an elbow in a number of clusters vs. intertia plot.

        Parameters:
        __________

            k_vs_intertia_df : (pd.DataFrame) - A pandas frame containing the interia and the number of clusters from Kmeans clustering

        Returns:
        ________

            n_clusters: (int) - the number of clusters where the elbow was located

        *** This code was adapted from code provided by Professor Steven Morin ***
    """
    # set hyperparameter values
    s = 1.0
    curve = 'convex'
    direction = 'decreasing'

    # call KneeLocator; attempt to find elbow
    kneedle = KneeLocator(k_vs_inertia_df.n_clusters, k_vs_inertia_df.inertia, S = s, curve = curve, direction = direction)

    try:
        n_clusters = kneedle.elbow
    except Exception as e:
        n_clusters = None

    return n_clusters

###########################################################################################################################
#####################################           get_best_eps_and_k           ##############################################
###########################################################################################################################
def get_best_eps_and_k(embedding: np.ndarray) -> tuple[float, int]:
    """
        Function: get_best_eps_and_k

            This function will best values of the eps and min samples hyperparmeters for DBscan

        Parameters:
        __________

            embedding : (np.ndarray) - the embedding generated by UMAP

        Returns:
        ________

            eps: (float) - the best eps value to use in DBscan
            min_samples: (int) - the min samples associated with the best eps, also to be used in DBscan

        *** This code was adapted from code provided by Professor Steven Morin ***
    """
    # instantiate KDTree
    kdt = KDTree(embedding)

    # iterate over min_samples
    min_samples_list = [3, 4, 5, 6, 7]
    df_row_dict_list = []
    closest_kth_dict = {}
    for k in min_samples_list:

        # generate list of nearest neighbors
        closest_kth = []
        for i in range(embedding.shape[0]):
            dist, _ = kdt.query(embedding[i, :].reshape(1, -1), k + 1)  
            closest_kth.append(dist[0, -1])

        # sort the distances and store in dictionary
        closest_kth_dict[k] = sorted(closest_kth)

        # set the kneed parameters to find elbow
        curve = 'convex'
        direction = 'increasing'
        drop_first_percent = 0.10
        drop_first = int(drop_first_percent * embedding.shape[0])
        y = closest_kth_dict[k][drop_first:] 

        # find the elbow and get eps and min_samples
        s = 1.0
        kneedle = KneeLocator(list(range(len(y))), y, S = s, curve = curve, direction = direction)
        idx = kneedle.elbow
        eps = y[idx]
        df_row_dict_list.append(
            {
                'index': drop_first + idx,
                'k': k,
                'eps': eps
            }
        )

    # create temp_df
    temp_df = pd.DataFrame(df_row_dict_list).sort_values('eps')

    # extract eps and k
    eps = temp_df["eps"].max()
    min_samples = temp_df.loc[temp_df.eps == eps, "k"].values[0]

    return eps, min_samples

###########################################################################################################################
#####################################           get_best_emedding_index          ##########################################
###########################################################################################################################
def get_best_embedding_index(results_df: pd.DataFrame) -> int:
    """
        Function: get_best_embedding_index

            This function will find the best index of the of the model with highest validity index or
                silhouette score and return the index

        Parameters:
        __________

            results_df : (pd.DataFrame) - results of grid search over UMAP hypers

        Returns:
        ________

            latent_manifold_details_index: (int) - the index of the that contains the embedding and model details
                corresponding to best embedding in the results_df

    """
    # generate list of algos in results df
    algo_list = [x for x in results_df.algo]

    # get max validity score in DBscan is in algos list
    if "DBScan" in algo_list:
        max_validity_index = results_df["validity_index"].idxmax()
        max_validity_score = results_df['validity_index'][max_validity_index]
    else:
        max_validity_score = 0

    # get max silhouette score in kmeans is in algos list
    if "kmeans" in algo_list:
        max_sil_index = results_df["silhouette_score"].idxmax()
        max_sil_score = results_df["silhouette_score"][max_sil_index]

    else:
        max_sil_score = 0

    # select the model with the higher score
    if max_validity_score > max_sil_score:
        latent_manifold_details_index = max_validity_index
    else:
        latent_manifold_details_index = max_sil_index

    return latent_manifold_details_index

###########################################################################################################################
#################################     get_best_cluster_label_permutation      #############################################
###########################################################################################################################
def get_best_cluster_label_permutation(labels_pred, labels_true, print_out=False, duration_in_seconds = 30):
    """
        Function: get_best_cluster_label_permutaion

            This function will scan all the predicted label permutation searching of the one that maximizes
                the trace of the contingency matrix. 

        Parameters:
        __________

            labels_pred : (np.ndarray) - array of cluster labels
            labels_true : (np.ndarray) - array of true labels
            print_out : (bool) - flag determining printing contingency matrix and trace at each iteration
            duration_in_second : (int) - how long to performt the search

        Returns:
        ________

            results_dict : (dict) - dict containing contingency matrix, mapping, interations, number of permutations

        *** This code was adapted by code provided by Professor Steven Morin ***
    """
    # set the stop time for now + duration
    stop_time = time.time() + duration_in_seconds

    # declare variables
    best_permutation_mapping = None
    best_perm_labels_pred = None
    best_contingency_matrix = None
    iters = 0
    max_contingency_matrix_trace = 0
    permutations = itertools.permutations(set(labels_pred))

    for cluster_labels in permutations:

        mapping = dict(zip(set(labels_pred), cluster_labels))
        perm_labels_pred = [mapping[label] for label in labels_pred]

        contingency_matrix_ = contingency_matrix(labels_true, perm_labels_pred)

        contingency_matrix_trace = np.trace(contingency_matrix_)

        if print_out:
            print(f'\ncontingency_matrix:\n{contingency_matrix_}')
            print(f'\ncontingency_matrix_trace:\n{contingency_matrix_trace}')

        if contingency_matrix_trace > max_contingency_matrix_trace:
            max_contingency_matrix_trace = contingency_matrix_trace
            best_permutation_mapping = mapping
            best_perm_labels_pred = perm_labels_pred
            best_contingency_matrix = contingency_matrix_
        
        # update iterations
        iters += 1
    
        if time.time() >= stop_time:
            break
  
    return {
        'best_permutation_mapping': best_permutation_mapping,
        'best_perm_labels_pred': best_perm_labels_pred,
        'best_contingency_matrix': best_contingency_matrix,
        'num_iterations': iters,
        'max_iters': permutations
    }

###########################################################################################################################
##############################     remove_noise_data_objects_from_labels      #############################################
###########################################################################################################################
def remove_noise_data_objects_from_labels(labels_pred: np.ndarray, labels_true: np.ndarray) -> tuple:
    """
        Function: remove_noise_data_objects_from_labels

            This function will remove all the data points labeled as noice

        Parameters:
        __________

            labels_pred : (np.ndarray) - array of cluster labels
            labels_true : (np.ndarray) - array of true labels

        Returns:
        ________

            labels_pred : (np.ndarray) - array of cluster labels
            labels_true : (np.ndarray) - array of true labels

        *** This code was provided by Professor Steven Morin ***
    """
    labels_pred = np.array(labels_pred).reshape(-1, 1)
    labels_true = np.array(labels_true).reshape(-1, 1)
    labels = np.concatenate((labels_pred, labels_true), axis = 1)

    labels = labels[~np.any(labels == -1, axis = 1), :]

    true_labels = labels[:, 0]
    pred_labels = labels[:, 1]

    return true_labels, pred_labels

###########################################################################################################################
########################################       get_adjusted_rand_score        #############################################
###########################################################################################################################
def get_adjusted_rand_index(results_df: pd.DataFrame, true_labels_df: pd.DataFrame, best_model_index: int) -> float:
    """
        Function: get_adjusted_rand_index

            This function will find the adjusted rand score

        Parameters:
        __________

            results_df : (pd.DataFrame) - results of grid search over UMAP hypers
            true_labels_df : (pd.DataFrame) - a dataframe containing the true labels of the data used in UMAP grid search.

        Returns:
        ________

            adj_rand_score : (float) - the adjusted rand score of the best clustering algorithm.
    """

    # check which algo generated the best clustering
    if results_df.loc[best_model_index, "algo"] == "DBScan":

        # get label arrays
        predicted_labels = np.array(results_df.loc[best_model_index, "cluster_labels"])
        true_labels = np.array(true_labels_df['label'])

        # remove noise
        true_labels, predicted_labels = remove_noise_data_objects_from_labels(predicted_labels, true_labels)

    # kmeans
    else:

        predicted_labels = np.array(results_df.loc[best_model_index, "cluster_labels"])
        true_labels = np.array(true_labels_df['label'])

    # get the adjusted rand score
    adj_rand_score = adjusted_rand_score(predicted_labels, true_labels)

    return adj_rand_score

###########################################################################################################################
########################################     get_best_mapping_using_mode      #############################################
###########################################################################################################################
def get_best_mapping_using_mode(labels_and_image_data_df: pd.DataFrame) -> dict:
    """
        Function: get_best_mapping_using_mode

            This function will find the best mapping of true labels to predicted labels. It will do this by finding mode of each
            predicted label and assigning it to corresponding true label.

        Parameters:
        __________

            labels_and_image_data_df: (pd.DataFrame) - dataframe that contains true and predicted labels columns (label, predicted_labels).

        Returns:
        ________

            remap_cluster_labels: (dict) - a dictionary of the best mapping of predicted labels to true labels

        *** This code was provided by Professor Steven Morin ***

    """
    array = labels_and_image_data_df[['label', 'predicted_labels']].values
    df_row_dict_list = []
    for i in labels_and_image_data_df.predicted_labels.unique():

        sub_array = array[array[:, 1] == i]
        mode_ = mode(sub_array[:, 0]).mode

        value_counts = labels_and_image_data_df.loc[labels_and_image_data_df.predicted_labels == i, 'label'].value_counts()

        df_row_dict_list.append(
            {
                'cluster_label': i,
                'cluster_mode': mode_,
                'cluster_mode_count': value_counts.loc[mode_],
                'cluster_purity': value_counts.loc[mode_]/sub_array.shape[0]
            }
        )
        
    results_df = pd.DataFrame(df_row_dict_list)

    results_df.sort_values(['cluster_mode', 'cluster_mode_count'], ascending=False)

    cluster_label_list = list(labels_and_image_data_df.predicted_labels.unique())
    class_label_list = list(labels_and_image_data_df.label.unique())
    remap_cluster_labels = {}
    len_class_label_list = len(class_label_list)
    i = 0
    for index, row in results_df.sort_values(['cluster_mode', 'cluster_mode_count'], ascending=False).iterrows():
        if row['cluster_label'] in cluster_label_list and row['cluster_mode'] in class_label_list:
            remap_cluster_labels[row['cluster_label']] = row['cluster_mode']
            cluster_label_list.remove(row['cluster_label'])
            class_label_list.remove(row['cluster_mode'])
        else:
            i += 1
            remap_cluster_labels[row['cluster_label']] = len_class_label_list + i
            cluster_label_list.remove(row['cluster_label'])

    return remap_cluster_labels

###########################################################################################################################
###########################################       get_class_purity       ##################################################
###########################################################################################################################
def get_class_purity(contingency_matrix: np.ndarray) -> pd.DataFrame:
    """
        Function: get_class_purity

            This function will find the purity of each true label with respect to predicted label.

        Parameters:
        __________

            contingency_matrix: (np.ndarray) - a contingency matrix

        Returns:
        ________

            purity_df: (pd.DataFrame) - a dataframe containing the purity information for each true label
    """

    # put c matrix into a dataframe
    contingency_matrix_df = pd.DataFrame(contingency_matrix)

    # extract the sum along the rows, true label totals
    true_label_sums = np.array(contingency_matrix_df.sum(axis=1))

    # extract the trace, true predicted
    trace = np.diag(contingency_matrix_df.values)

    # compute purity
    purity = np.round((trace / true_label_sums), 5)

    # put result in a dataframe
    true_labels = np.arange(0, len(true_label_sums), 1)
    purity_dict = {"class_labels": true_labels,
                "class_count": true_label_sums,
                "predicted_correct": trace,
                "class_purity": purity}
    purity_df = pd.DataFrame(purity_dict)

    return purity_df

###########################################################################################################################
########################################       plot_mislabeled_examples       #############################################
###########################################################################################################################
def plot_mislabeled_examples(mislabeled_df, num_mislabeled, num_components, true_label, path):
    """
        Function: plot_mislabeled_examples

            This code will plot mislabeled image examples 

        Parameters:
        __________

            mislabeled_df : (pd.DataFrame) - dataframe containing mislabeled image pixel data
            num_mislabeled: (int) - the number of mislabeled images, i.e., the len(mislabeled_df)
            num_components : (int) - the number of components in the embedding space where clustering was performed
            path : (str) - location to save plot

        Returns:
        ________

            None
    """

    # if there are greater than 4 mislabeled in class, plot 4
    if num_mislabeled >= 4:
        fig = plt.figure(figsize = (6, 6))
        fig.suptitle(f'Mislabeled Image Examples', weight = "bold", fontsize = 20)
        for i in range(4):
            image_array = mislabeled_df.loc[i, '0':'63'].values.reshape(8, 8)
            plt.subplot(2, 2, i + 1)
            plt.tight_layout()
            plt.imshow(image_array, cmap = "gray", interpolation = 'none')
            plt.text(.5, 1.14, 
                f'True Label: { mislabeled_df.loc[i, "label"] }', 
                fontsize = 11, 
                ha='center', 
                va='bottom', 
                transform = plt.gca().transAxes,
                color = 'blue',
                weight = 'bold')
            plt.text(.5, 1.02, 
                f'Predicted Label: {mislabeled_df.loc[i, "predicted_labels"]}', 
                fontsize = 10, 
                ha = 'center', 
                va = 'bottom', 
                transform = plt.gca().transAxes,
                color = "red",
                weight = 'bold',
                style = 'italic')
            plt.xticks([])
            plt.yticks([])

    # if there are 3, plot 3
    elif num_mislabeled == 3:
        fig = plt.figure(figsize = (6, 4))
        fig.suptitle(f'Mislabeled Image Examples', weight = "bold", fontsize = 20, y = 0.95)
        for i in range(3):
            image_array = mislabeled_df.loc[i, '0':'63'].values.reshape(8, 8)
            plt.subplot(1, 3, i + 1)
            plt.tight_layout()
            plt.imshow(image_array, cmap = "gray", interpolation = 'none')
            plt.text(0.5, 1.15, 
                f'True Label: { mislabeled_df.loc[i, "label"] }', 
                fontsize = 11, 
                ha='center', 
                va='bottom', 
                transform = plt.gca().transAxes,
                color = 'blue',
                weight = 'bold')
            plt.text(0.5, 1.02, 
                f'Predicted Label: {mislabeled_df.loc[i, "predicted_labels"]}', 
                fontsize = 10, 
                ha = 'center', 
                va = 'bottom', 
                transform = plt.gca().transAxes,
                color = "red",
                weight = 'bold',
                style = 'italic')
            plt.xticks([])
            plt.yticks([])

    # if there are 2, plot 2
    elif num_mislabeled == 2:
        fig = plt.figure(figsize = (5, 4))
        fig.suptitle(f'Mislabeled Image Examples', weight = "bold", fontsize = 20, y = .98)
        for i in range(2):
            image_array = mislabeled_df.loc[i, '0':'63'].values.reshape(8, 8)
            plt.subplot(1, 2, i + 1)
            plt.tight_layout()
            plt.imshow(image_array, cmap = "gray", interpolation = 'none')
            plt.text(0.5, 1.15, 
                f'True Label: { mislabeled_df.loc[i, "label"] }', 
                fontsize = 11, 
                ha='center', 
                va='bottom', 
                transform = plt.gca().transAxes,
                color = 'blue',
                weight = 'bold')
            plt.text(0.5, 1.02, 
                f'Predicted Label: {mislabeled_df.loc[i, "predicted_labels"]}', 
                fontsize = 10, 
                ha = 'center', 
                va = 'bottom', 
                transform = plt.gca().transAxes,
                color = "red",
                weight = 'bold',
                style = 'italic')
            plt.xticks([])
            plt.yticks([])

    # if there are 1, plot 1
    elif num_mislabeled == 1:
        fig = plt.figure(figsize = (4, 3))
        image_array = mislabeled_df.loc[0, '0':'63'].values.reshape(8, 8)
        plt.imshow(image_array, cmap = "gray", interpolation = 'none')
        plt.text(0.5, 1.32, 
            f'True Label: { mislabeled_df.loc[0, "label"] }', 
            fontsize = 11, 
            ha='center', 
            va='bottom', 
            transform = plt.gca().transAxes,
            color = 'blue',
            weight = 'bold')
        plt.text(0.5, 1.02, 
            f'Predicted Label: {mislabeled_df.loc[0, "predicted_labels"]}', 
            fontsize = 10, 
            ha = 'center', 
            va = 'bottom', 
            transform = plt.gca().transAxes,
            color = "red",
            weight = 'bold',
            style = 'italic')
        plt.title(f'Mislabeled Image Example', weight = "bold", fontsize = 20, y = 1.7)                               
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
    
    # save fig
    plt.savefig(path + f"/figs/n_{num_components}_true_label_{true_label}.png", bbox_inches = 'tight')
    plt.show();

    return None

###########################################################################################################################
###################################       plot_erroneous_label_examples       #############################################
###########################################################################################################################
def plot_erroneous_label_examples(labels_and_image_data_df, erroneous_labels_list, num_components, save_path):
    """
        Function: plot_mislabeled_examples

            This code will plot images whose label is not present in the true labels set 

        Parameters:
        __________

            labels_and_image_data_df : (pd.DataFrame) - dataframe containing cluster labels (predicted_labels), true labels (label), 
                and image data (0, 64).
            erronesous_labels_list: (list) - a list of cluster labels not present in true labels set 
            num_components : (int) - the number of components in the embedding space where clustering was performed
            save_path : (str) - location to save plot

        Returns:
        ________

            None
    """

    # iterate over erroneous labels
    for i in erroneous_labels_list:
        filtered_df = labels_and_image_data_df[labels_and_image_data_df["predicted_labels"] == i]

        filtered_df.reset_index(inplace = True)
        print('-' * 125)
        print(f"Example Images from Predicted Label: {i}")
        fig = plt.figure(figsize = (10, 10))
        fig.suptitle(f'Erroneous Label: {i}', weight = "bold", fontsize = 20)
        num_images = len(filtered_df) if len(filtered_df) <= 16 else 16
        for j in range(num_images):
            image_array = filtered_df.loc[j, '0':'63'].values.reshape(8, 8)
            plt.subplot(4, 4, j + 1)
            plt.tight_layout()
            plt.imshow(image_array, cmap = "gray", interpolation = 'none')
            plt.text(.5, 1.14, 
                f'True Label: { filtered_df.loc[j, "label"] }', 
                fontsize = 11, 
                ha='center', 
                va='bottom', 
                transform = plt.gca().transAxes,
                color = 'blue',
                weight = 'bold')
            plt.text(.5, 1.02, 
                f'Predicted Label: {filtered_df.loc[j, "predicted_labels"]}', 
                fontsize = 10, 
                ha = 'center', 
                va = 'bottom', 
                transform = plt.gca().transAxes,
                color = "red",
                weight = 'bold',
                style = 'italic')
            plt.xticks([])
            plt.yticks([])

        # save fig
        plt.savefig(save_path + f"/figs/n_{num_components}_erroneous_label_{i}.png", bbox_inches = 'tight')
        plt.show();

    return None

###########################################################################################################################
###################################   plot_2D_embedding_with_cluster_label_results    #####################################
###########################################################################################################################
def plot_2D_embedding_with_cluster_label_results(labels_and_image_data_df, algo, validity_index, adjusted_rand, path):
    """
        Function: plot_mislabeled_examples

            This code will plot predicted labels and mislabeled elements with respect to a 2D embedding.

        Parameters:
        __________

            labels_and_image_data_df : (pd.DataFrame) - dataframe containing cluster labels (predicted_labels), true labels (label), 
                image data (0, 64), and embedding data (x, y).
            algo: (str) - the algorithm used in clustering 
            validity_index : (float) - the validity index of the clustering solution.
            adjusted_rand : (float) - the adjusted rand score index of the clustering solution.
            path : (str) - location to save plot

        Returns:
        ________

            None
    """
    plt.figure(figsize=(10, 7))
    plt.scatter(labels_and_image_data_df.x, labels_and_image_data_df.y, 
            c = labels_and_image_data_df.predicted_labels, 
            cmap='Spectral', 
            s = 5)

    plt.gca().set_aspect('equal', 'datalim')

    labels = np.unique(labels_and_image_data_df.predicted_labels)
    num_colors = len(labels)
    color_bar = plt.colorbar(boundaries=np.arange(num_colors + 1) - 0.5, ticks=np.arange(num_colors))
    color_bar.set_ticklabels(labels)

    plt.scatter(labels_and_image_data_df[labels_and_image_data_df.label != labels_and_image_data_df.predicted_labels].x, 
            labels_and_image_data_df[labels_and_image_data_df.label != labels_and_image_data_df.predicted_labels].y, 
            marker = 'x', 
            c = 'black', 
            s = 20,
            label = "Mislabeled")
    plt.text(0.5, 1.19, 
            '2D Embedding Space', 
            fontsize = 18, 
            ha='center', 
            va='bottom', 
            transform=plt.gca().transAxes, 
            weight = 'bold')
    plt.text(0.5, 1.13, 
            'Predicted Labels ~ Mislabeled Highlight', 
            fontsize = 16, 
            ha='center', 
            va='bottom', 
            transform=plt.gca().transAxes, 
            weight = 'bold')
    plt.text(0.5, 1.09, 
            f'Algorithm = {algo}', 
            ha ='center', 
            va='bottom', 
            transform=plt.gca().transAxes, 
            fontsize = 12, 
            weight = 'bold', 
            style = 'italic')
    plt.text(0.5, 1.05,
            f'Validity Index = {validity_index:.5f}',
            ha = 'center', 
            va='bottom', 
            transform=plt.gca().transAxes, 
            fontsize = 12,  
            weight = 'bold', 
            style = 'italic')
    plt.text(0.5, 1.01,
            f'Adjusted Rand Score = {adjusted_rand:.5f}', 
            ha = 'center', 
            va = 'bottom', 
            transform=plt.gca().transAxes, 
            fontsize = 12,  
            weight = 'bold', 
            style = 'italic', 
            color = 'red')
    plt.legend()

    # save fig
    plt.savefig(path + "/figs/embedding_mislabeled_hightlight.png", bbox_inches = 'tight')
    plt.show();

