'''
    Midterm: Clustering - Digits Dataset
    Joseph Nelson Farrell
    DS 5230 Unsupervised Machine Learning
    Northeastern University
    Steven Morin, PhD

   This file contains a function that perform dimensionality reduction using UMAP.
   The function will return a results dict with hyperparameter values, the embedding, 
   and the trustworthiness score.
'''

from sklearn.manifold import trustworthiness
import pandas as pd
import umap

def umap_dim_red(cap_x_df: pd.DataFrame, n_neighbors: int, min_dist: float, metric: str, n_components: int) -> dict:
    """
        Function: umap_dim_red

        This function will perfrom UMAP dimensionality reduction and return a dictionary with the results.

        Parameters:
        ___________

        cap_x_df : (pd.DataFrame) containing the design matrix

        n_neighbors : (int) the number of neighbors hyperparameter for UMAP

        min_dist : (float) the minimum distance hyperparameter for UMAP

        metric : (str) the distance metrix hyperparameter for UMAP

        n_componenents : (int) the number of components hyperparameter for UMAP

        Returns:
        ________

        resutlts_dict: (dict) A dictionary containing the embedding, the trustworthiness score and all the 
            the hyperparamers used.
    
    """
    # instantiate results dict
    results_dict = {"embedding": [],
                    "n_neighbors":[],
                    "min_dist": [],
                    "metric": [],
                    "n_components": [],
                    "trustworthiness": [],
    }

    # instantiate reducer
    reducer = umap.UMAP(
                        n_neighbors = n_neighbors, 
                        n_components = n_components, 
                        metric = metric, 
                        min_dist = min_dist, 
                        spread = 1.0, 
                        random_state = 42
                                            )
    # fit reducer, get embedding
    reducer.fit(cap_x_df)
    embedding = reducer.transform(cap_x_df)

    # compute trustworthiness
    trust_score = trustworthiness(cap_x_df, embedding, n_neighbors = n_neighbors, metric = metric)

    # update results dict
    results_dict["n_neighbors"].append(n_neighbors)
    results_dict["n_components"].append(n_components)
    results_dict["min_dist"].append(min_dist)
    results_dict["trustworthiness"].append(trust_score)
    results_dict['embedding'].append(embedding)
    results_dict['metric'].append(metric)

    return results_dict