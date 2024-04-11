from sklearn.datasets import make_moons
from sklearn.datasets import make_blobs
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import utils.clustering_utils as clu


def make_moons_data_set(n_samples=1000, noise=0.05, random_state=42, print_out=True, plot_data=True):

    cap_x, y = make_moons(n_samples=n_samples, shuffle=True, noise=noise, random_state=random_state)

    if print_out:
        print(f'\n', '*'*80, sep='')
        print(f'make moons data set\n')
        print(f'data set as np arrays:\n')
        print(f'cap_x.shape: {cap_x.shape}')
        print(f'cap_x[0:10, :]:\n{cap_x[0:5, :]}')
        print(f'\ny.shape: {y.shape}')
        print(f'y[0:5]:\n{y[0:5]}')
        print(f'np.unique(y): {np.unique(y)}')

    cap_x_df_columns = ['x' + str(i + 1) for i in range(cap_x.shape[1])]
    cap_x_y_df = pd.DataFrame(
        data=np.concatenate((cap_x, y.reshape(-1, 1)), axis=1),
        index=range(cap_x.shape[0]),
        columns=cap_x_df_columns + ['y']
    )
    cap_x_y_df.y = cap_x_y_df.y.astype(int)

    if print_out:
        print(f'\ndata set as data frame:\n')
        print(f'cap_x_y_df.shape: {cap_x_y_df.shape}')
        print(f'cap_x_y_df.head():\n{cap_x_y_df.head()}')

    if plot_data:
        ax = sns.scatterplot(data=cap_x_y_df, x='x1', y='x2')
        ax.set_aspect('equal', adjustable='box')
        plt.title(f'n_samples: {n_samples}; noise: {noise}')
        plt.grid()
        plt.show()

    return cap_x, y, cap_x_y_df


def make_blobs_data_set(centers, cluster_std, n_samples=1000, random_state=42, print_out=True, plot_data=True):

    cap_x, y = make_blobs(n_samples=n_samples, centers=centers, cluster_std=cluster_std, random_state=random_state)

    if print_out:
        print(f'\n', '*'*80, sep='')
        print(f'make blobs data set\n')
        print(f'data set as np arrays:\n')
        print(f'cap_x.shape: {cap_x.shape}')
        print(f'cap_x[0:10, :]:\n{cap_x[0:5, :]}')
        print(f'\ny.shape: {y.shape}')
        print(f'y[0:5]:\n{y[0:5]}')
        print(f'np.unique(y): {np.unique(y)}')

    cap_x_df_columns = ['x' + str(i + 1) for i in range(cap_x.shape[1])]
    cap_x_y_df = pd.DataFrame(
        data=np.concatenate((cap_x, y.reshape(-1, 1)), axis=1),
        index=range(cap_x.shape[0]),
        columns=cap_x_df_columns + ['y']
    )
    cap_x_y_df.y = cap_x_y_df.y.astype(int)

    if print_out:
        print(f'\ndata set as data frame:\n')
        print(f'cap_x_y_df.shape: {cap_x_y_df.shape}')
        print(f'cap_x_y_df.head():\n{cap_x_y_df.head()}')

    if plot_data:
        ax = sns.scatterplot(data=cap_x_y_df, x='x1', y='x2')
        ax.set_aspect('equal', adjustable='box')
        plt.title(f'n_samples: {n_samples}')
        plt.grid()
        plt.show()

    return cap_x, y, cap_x_y_df


def get_randomly_distributed(n_samples):

    # get a data set from moons to use as a template
    cap_x, y, cap_x_y_df = make_moons_data_set(n_samples=n_samples, noise=0.10)

    # get the equivalent randomly distributed data set
    cap_x = clu.get_randomly_distributed_data(cap_x, seed=42, plots=True)

    # package it up
    cap_x_df_columns = ['x' + str(i + 1) for i in range(cap_x.shape[1])]
    cap_x_y_df = pd.DataFrame(
        data=cap_x,
        index=range(cap_x.shape[0]),
        columns=cap_x_df_columns
    )

    return cap_x, None, cap_x_y_df


def get_regularly_distributed(n_samples, data_set_dict):

    # get a data set from moons to use as a template
    cap_x, y, cap_x_y_df = make_moons_data_set(n_samples=n_samples, noise=0.10)

    # now form the grid of regularly spaced points
    data_max = cap_x.max(axis=0)
    data_min = cap_x.min(axis=0)

    x_spread = data_max[0] - data_min[0]
    y_spread = data_max[1] - data_min[1]
    x_y_spread_ratio = x_spread/y_spread

    n_samples = cap_x.shape[0]
    n_y = int(round(np.sqrt(n_samples/x_y_spread_ratio), 0))
    n_x = int(round(x_y_spread_ratio * np.sqrt(n_samples/x_y_spread_ratio), 0))

    x = np.arange(data_min[0], data_max[0], x_spread/n_x)
    y = np.arange(data_min[1], data_max[1], y_spread/n_y)

    x, y = np.meshgrid(x, y, sparse=False)
    cap_x = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), axis=1)

    # make data frame
    cap_x_df_columns = ['x' + str(i + 1) for i in range(cap_x.shape[1])]
    cap_x_y_df = pd.DataFrame(
        data=cap_x,
        index=range(cap_x.shape[0]),
        columns=cap_x_df_columns
    )

    # package it up
    data_set_dict['regularly_distributed'] = \
        {
            'cap_x': cap_x,
            'y': None,
            'cap_x_y_df': cap_x_y_df,
            'spec_type': None,
            'spec_value': np.nan,
            'data_set_type': 'regularly_distributed'
        }

    return data_set_dict


def get_well_centered(n_samples, data_set_dict, x_step_size, x_start, cluster_std=(0.1, 0.1, 0.1), name_extra=None):

    if name_extra is None:
        name_extra = ''

    for x_ in np.arange(x_start, -1.0, x_step_size):
        print(x_)
        centers = np.array([[x_, 0.5 * x_ + 0.5], [x_, -0.5 * x_ - 0.5], [-1.0, 0.0]])
        cluster_std = np.array([cluster_std[0], cluster_std[1], cluster_std[2]])
        cap_x, y, cap_x_y_df = make_blobs_data_set(centers, cluster_std, n_samples=n_samples)
        separation = np.linalg.norm(np.array([x_, -0.5 * x_ - 0.5]) - np.array([-1.0, 0.0])).round(2)
        data_set_dict['well_separated_' + name_extra + str(separation).replace('.', '_')] = \
            {
                'cap_x': cap_x,
                'y': y,
                'cap_x_y_df': cap_x_y_df,
                'spec_type': 'separation',
                'spec_value': separation,
                'data_set_type': 'well_separated'
            }

    return data_set_dict


def make_data_sets_to_demo_cluster_validation(noisy_moons=True, well_separated=True, center_based=True,
                                              randomly_distributed=False, well_separated_real_tight=False,
                                              regularly_distributed=False):

    data_set_dict = {}
    n_samples = 1000

    if noisy_moons:
        # make noisy moons clusters
        for noise in [0.05, 0.075, 0.10, 0.15]:
            cap_x, y, cap_x_y_df = make_moons_data_set(n_samples=n_samples, noise=noise)
            data_set_dict['moons_noise_' + str(noise).replace('.', '_')] = \
                {
                    'cap_x': cap_x,
                    'y': y,
                    'cap_x_y_df': cap_x_y_df,
                    'spec_type': 'noise',
                    'spec_value': noise,
                    'data_set_type': 'moons'
                }

    if well_separated:
        # make well separated clusters - bring well separated clusters closer together, so they are not so well
        # separated
        data_set_dict = get_well_centered(n_samples, data_set_dict, 0.1, -2.0)

    if center_based:
        # make center-based clusters
        centers = np.array([[-1.5, 2.25], [-1.0, 2.25]])
        cluster_std = np.array([0.1, 0.1])
        cap_x, y, cap_x_y_df = make_blobs_data_set(centers, cluster_std, n_samples=n_samples)
        data_set_dict['center_based'] = \
            {
                'cap_x': cap_x,
                'y': y,
                'cap_x_y_df': cap_x_y_df,
                'spec_type': None,
                'spec_value': np.nan,
                'data_set_type': 'center_based'
            }

    if randomly_distributed:
        # make randomly distributed data
        cap_x, y, cap_x_y_df = get_randomly_distributed(n_samples)
        data_set_dict['randomly_distributed'] = \
            {
                'cap_x': cap_x,
                'y': y,
                'cap_x_y_df': cap_x_y_df,
                'spec_type': None,
                'spec_value': np.nan,
                'data_set_type': 'randomly_distributed'
            }

    if well_separated_real_tight:
        data_set_dict = get_well_centered(n_samples, data_set_dict, 0.1, -2.0, cluster_std=(0.01, 0.01, 0.01),
                                          name_extra='tight')

    if regularly_distributed:
        data_set_dict = get_regularly_distributed(n_samples, data_set_dict)

    return data_set_dict


def make_clustering_data_sets_external_indices():

    data_set_dict = dict()

    # perfect clustering
    data_set_dict['perfect_clustering_1'] = {
        'labels_true': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 3, 1, 1, 3, 3, 3],
        'labels_pred': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 3, 1, 1, 3, 3, 3]
    }

    # more clusters than classes
    data_set_dict['more_clusters_than_classes'] = {
        'labels_true': [0, 0, 0, 1, 1, 1],
        'labels_pred': [0, 0, 1, 1, 2, 2]
    }

    # more classes than clusters
    data_set_dict['more_classes_than_clusters'] = {
        'labels_true': [0, 0, 0, 1, 2, 2],
        'labels_pred': [0, 0, 1, 1, 0, 1]
    }

    # okay clustering
    data_set_dict['okay_clustering_1'] = {
        'labels_true': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 3, 1, 1, 3, 3, 3],
        'labels_pred': [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
    }

    # random clustering
    size = 60
    np.random.seed(42)
    data_set_dict['random_clustering_1'] = {
        'labels_true': np.random.choice(a=[0, 1, 2], size=size, replace=True, p=None),
        'labels_pred': np.random.choice(a=[0, 1, 2, 3, 4], size=size, replace=True, p=None)
    }

    # kumar table 7.10 data set
    data_set_dict['kumar_table_7_10'] = {
        'labels_true': [0, 0, 1, 1, 1],
        'labels_pred': [0, 0, 0, 1, 1]
    }

    return {
        'data_set_dict': data_set_dict
    }


if __name__ == "__main__":
    pass
