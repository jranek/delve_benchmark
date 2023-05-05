import pandas as pd
import numpy as np
import os
import scanpy as sc 
import anndata
import delve_benchmark
import scprep
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from scipy.stats import kendalltau

def splatter_sim(cells_per_path = 250,
                n_paths = 6,
                n_genes = 500,
                bcv_common = 0.1,
                lib_loc = None,
                dropout_type = 'none',
                dropout_prob = 0,
                path_from = None,
                path_skew = None,
                path_type = None,
                group_prob = None,
                random_state = 0):
    """Simulates a single-cell RNA sequencing trajectory using Splatter: https://genomebiology.biomedcentral.com/articles/10.1186/s13059-017-1305-0. 
    ~~~ Uses the scprep wrapper function: https://scprep.readthedocs.io/en/stable/_modules/scprep/run/splatter.html ~~~  
    Parameters
    For more details on the parameters, see: https://scprep.readthedocs.io/en/stable/_modules/scprep/run/splatter.html#SplatSimulate
    ----------
    Returns
    adata: anndata.AnnData
        annotated data object containing simulated single-cell RNA sequecing data (dimensions = cells x features)
    ----------
    """ 
    #set simulation parameters from real single-cell RNA sequencing dataset: https://pubmed.ncbi.nlm.nih.gov/27419872/
    params = {}
    params['group_prob'] = group_prob
    params['bcv_common'] = bcv_common
    params['path_from'] = path_from
    params['path_skew'] = path_skew

    params['mean_rate'] = 0.0173
    params['mean_shape'] = 0.54
    if lib_loc is None:
        params['lib_loc'] = 12.6
    else: 
        params['lib_loc'] = lib_loc
    params['lib_scale'] = 0.423
    params['out_prob'] = 0.000342
    params['out_fac_loc'] = 0.1
    params['out_fac_scale'] = 0.4
    params['bcv_df'] = 90.2

    results = scprep.run.SplatSimulate(method = 'paths', 
                                        batch_cells = [cells_per_path * n_paths], 
                                        group_prob = params['group_prob'], 
                                        n_genes = n_genes,
                                        de_prob = 0.1,
                                        de_down_prob = 0.5,
                                        de_fac_loc = 0.1,
                                        de_fac_scale = 0.4, 
                                        bcv_common = params['bcv_common'],
                                        dropout_type = 'none',
                                        path_from = params['path_from'],
                                        path_skew = params['path_skew'],
                                        mean_rate = params['mean_rate'],
                                        mean_shape = params['mean_shape'],
                                        lib_loc = params['lib_loc'], 
                                        lib_scale = params['lib_scale'], 
                                        out_prob = params['out_prob'], 
                                        out_fac_loc = params['out_fac_loc'], 
                                        out_fac_scale = params['out_fac_scale'], 
                                        bcv_df = params['bcv_df'],
                                        seed = random_state)

    data = pd.DataFrame(results['counts'])
    group = results['group'].copy()
    metadata = pd.DataFrame({'group':group.astype('str'), 'step':results['step'].astype(int)})

    if path_type == 'linear':
        metadata = linear_mask_(metadata)
    elif path_type == 'branch':
        metadata = branch_mask_(metadata)

    de_genes = pd.concat([pd.DataFrame(results['de_fac_1'], columns = ['path1']),
                            pd.DataFrame(results['de_fac_2'], columns = ['path2']),
                            pd.DataFrame(results['de_fac_3'], columns = ['path3']),
                            pd.DataFrame(results['de_fac_4'], columns = ['path4']),
                            pd.DataFrame(results['de_fac_5'], columns = ['path5']),
                            pd.DataFrame(results['de_fac_6'], columns = ['path6'])], axis = 1)
                            
    gene_index = []
    for i in range(0, len(de_genes.index)):
        if de_genes.loc[i].sum() != n_paths:
            id = 'DE_group_' + '_'.join(map(str, (np.where(de_genes.loc[i] !=1)[0]))) + '.{}'.format(i)
            gene_index.append(id)
        else:
            gene_index.append(str(i))

    cell_index = pd.Index(['cell_{}'.format(i) for i in range(metadata.shape[0])])
    data.index = cell_index
    data.columns = gene_index
    metadata.index = cell_index

    adata = anndata.AnnData(data)
    adata.obs = metadata

    if dropout_type == 'sim_binom':
         adata = sim_binom(adata = adata, dropout_prob = dropout_prob, random_state = random_state)

    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    return adata

def run_splatter(cells_per_path = 250,
                n_paths = 6,
                n_genes = 500,
                feature_threshold = 100,
                k_sweep = [25, 50, 100, 150, 200],
                n_trials = 10,
                bcv_common = 0.1,
                lib_loc = None,
                dropout_type = 'none',
                dropout_prob = 0.5,
                path_from = None,
                path_skew = None,
                path_type = None,
                labels_key = 'group',
                random_state = 0,
                save_directory = None):
    """Performs evaluation of simulated single-cell RNA sequencing data across feature selection methods
            1. single-cell RNA sequencing datasets are simulated using Splatter with defined trajectory structure
            2. noise corruption is added
            3. feature selection is performed and evaluated according to: precision@k, kNN classification accuracy, correlation of estimated pseudotime to ground truth 
    ----------
    """ 
    np.random.seed(random_state)
    k = 10
    n_pcs = 50
    num_subsamples = 1000
    n_clusters = 3
    n_random_state = 10
    pak_scores = []
    acc_scores = []
    kt_scores = []
    for trial in range(0, n_trials):
        group_prob = np.random.dirichlet(np.ones(n_paths) * 1.).round(3)
        group_prob = _sum_to_one(group_prob)

        path_skew = np.random.beta(10., 10., n_paths)

        adata = splatter_sim(cells_per_path = cells_per_path, n_paths = n_paths, n_genes = n_genes, bcv_common = bcv_common, lib_loc = lib_loc,
                            dropout_type = dropout_type, dropout_prob = dropout_prob, path_from = path_from, path_skew = path_skew,
                            group_prob = group_prob, path_type = path_type, random_state = random_state)

        reference_features = adata.var_names[np.where(adata.var_names.str.startswith('DE') == True)[0]] #ground truth differentially expressed features

        fs_methods = {'random_forest':{'n_splits': 3, 'labels_key': labels_key},
                        'laplacian_score_fs':{'k': k},
                        'neighborhood_variance_fs':{},
                        'mcfs_fs':{'n_selected_features': max(k_sweep), 'n_clusters':n_paths, 'k': k},
                        'scmer_fs':{'k': k, 'n_pcs': n_pcs},
                        'hotspot_fs':{'k': k, 'n_pcs': n_pcs},
                        'variance_score':{},
                        'all_features':{},
                        'hvg':{'n_top_genes':feature_threshold, 'log':False},
                        'delve_fs_trial0':{'num_subsamples': num_subsamples, 'n_clusters': n_clusters, 'k': k, 'random_state': 0, 'n_random_state': n_random_state},
                        'delve_fs_trial1':{'num_subsamples': num_subsamples, 'n_clusters': n_clusters, 'k': k, 'random_state': 42, 'n_random_state': n_random_state},
                        'delve_fs_trial2':{'num_subsamples': num_subsamples, 'n_clusters': n_clusters, 'k': k, 'random_state': 35, 'n_random_state': n_random_state},
                        'delve_fs_trial3':{'num_subsamples': num_subsamples, 'n_clusters': n_clusters, 'k': k, 'random_state': 2048, 'n_random_state': n_random_state},
                        'delve_fs_trial4':{'num_subsamples': num_subsamples, 'n_clusters': n_clusters, 'k': k, 'random_state': 3059, 'n_random_state': n_random_state},
                        'seed_features_trial0':{'num_subsamples': num_subsamples, 'n_clusters': n_clusters, 'k': k, 'random_state': 0, 'n_random_state': n_random_state},
                        'seed_features_trial1':{'num_subsamples': num_subsamples, 'n_clusters': n_clusters, 'k': k, 'random_state': 42, 'n_random_state': n_random_state},
                        'seed_features_trial2':{'num_subsamples': num_subsamples, 'n_clusters': n_clusters, 'k': k, 'random_state': 35, 'n_random_state': n_random_state},
                        'seed_features_trial3':{'num_subsamples': num_subsamples, 'n_clusters': n_clusters, 'k': k, 'random_state': 2048, 'n_random_state': n_random_state},
                        'seed_features_trial4':{'num_subsamples': num_subsamples, 'n_clusters': n_clusters, 'k': k, 'random_state': 3059, 'n_random_state': n_random_state},
                        'random_features_trial0':{'random_state': 0},
                        'random_features_trial1':{'random_state': 42},
                        'random_features_trial2':{'random_state': 35},
                        'random_features_trial3':{'random_state': 2048},
                        'random_features_trial4':{'random_state': 3059}
                        }

        le = LabelEncoder()
        y = list(adata.obs[labels_key].values)
        y_bin = np.asarray(le.fit_transform(y).astype('int'))
        X = adata.X.copy()

        #split
        cv = StratifiedKFold(n_splits = 3, shuffle = True, random_state = 0)

        acc_scores_ = []
        pak_scores_ = []
        kt_scores_ = []
        for train_ix, test_ix in cv.split(X, y_bin):
            pak_df = pd.DataFrame()
            acc_df = pd.DataFrame()
            kt_df = pd.DataFrame()
            for key, value in fs_methods.items():
                fs_method = key
                id = key
                if 'trial' in key:
                    fs_method = key.split('_trial')[0]

                fs_method = eval('delve_benchmark.tl.' + fs_method)
                fs_method_params = value.copy() 

                # split into training and test data
                X_train, X_test = X[train_ix, :], X[test_ix, :]
                y_train, y_test = y_bin[train_ix], y_bin[test_ix]

                print('performing {} selection on {} cells'.format(fs_method.__name__, len(train_ix)))

                # perform feature selection
                try:
                    predicted_features = fs_method(adata = adata[train_ix, :], X = X_train, feature_names = adata.var_names, n_jobs = -1, **fs_method_params)

                    if type(predicted_features) is not pd.core.frame.DataFrame:
                        predicted_features = pd.DataFrame(index = predicted_features) 

                    if len(predicted_features) == 0: #if no predicted features then return nans
                        pak_df_ = pd.DataFrame(len(k_sweep)*[np.nan], index = k_sweep, columns = [id])
                        pak_df = pd.concat([pak_df_, pak_df], axis = 1)
                        acc_df_ = pd.DataFrame([np.nan], index = ['acc'], columns = [id])
                        acc_df = pd.concat([acc_df_, acc_df], axis = 1)
                        kt_df_ = pd.DataFrame([np.nan], index = ['kt'], columns = [id])
                        kt_df = pd.concat([kt_df_, kt_df], axis = 1)
                    
                    else: #evaluate p@k
                        k_sweep_ = k_sweep.copy()
                        if fs_method.__name__ == 'all_features': #if all, then constant across k 
                            pak_df_ = delve_benchmark.tl.pak_all(predicted_features = list(predicted_features.index), reference_features = reference_features, k_sweep = k_sweep_)
                            pak_df_.columns = [id]
                            pak_df = pd.concat([pak_df_, pak_df], axis = 1)
                        
                        #compute pak at varying k for alternative methods
                        else:
                            print('computing pak score')
                            pak_df_ = delve_benchmark.tl.pak(predicted_features = list(predicted_features.index), reference_features = reference_features, k_sweep = k_sweep_)
                            pak_df_.columns = [id]
                            pak_df = pd.concat([pak_df_, pak_df], axis = 1)

                        #select top features (except for all features)
                        if (feature_threshold is not None) and (fs_method.__name__ != 'all_features'):
                            predicted_features = np.asarray(predicted_features.index[:feature_threshold])
                        else:
                            predicted_features = np.asarray(predicted_features.index)

                        #subset X according to selected features
                        X_train = X_train[:, np.isin(adata.var_names, predicted_features)] 
                        X_test = X_test[:, np.isin(adata.var_names, predicted_features)]

                        #train knn classifier and predict labels
                        print('performing classification with 3 knn')
                        classifier = KNeighborsClassifier(n_neighbors = 3)
                        classifier.fit(X_train, y_train)
                        y_pred = classifier.predict(X_test)

                        #return accuracy score
                        acc = accuracy_score(y_true = y_test, y_pred = y_pred)
                        acc_df_ = pd.DataFrame([acc], index = ['acc'], columns = [id])
                        acc_df = pd.concat([acc_df_, acc_df], axis = 1)

                        #perform trajectory inference and compute correlation to ground truth simulated progression
                        if len(predicted_features) <= 2: #if no features, return nan
                            kt_df_ = pd.DataFrame([np.nan], index = ['kt'], columns = [id])
                            kt_df = pd.concat([kt_df_, kt_df], axis = 1)
                        else: 
                            adata_pseudo = adata[train_ix, np.isin(adata.var_names, predicted_features)].copy()
                            root_cell = np.where(adata_pseudo.obs['step'].values == min(adata_pseudo.obs['step'].values))[0][0] #root cells is the youngest ground truth cell

                            if len(adata_pseudo.var_names) < 20:
                                n_dcs = len(adata_pseudo.var_names)
                            else:
                                n_dcs = 20

                            adata_pseudo, pseudotime = delve_benchmark.tl.perform_dpt(adata = adata_pseudo, k = 10, n_dcs = n_dcs, root = root_cell)
                            kt_corr, _ = kendalltau(pseudotime, adata_pseudo.obs['step'].values)
                            kt_df_ = pd.DataFrame([kt_corr], index = ['kt'], columns = [id])
                            kt_df = pd.concat([kt_df_, kt_df], axis = 1)
                except ValueError:
                    continue
            
            pak_scores_.append(pak_df)
            acc_scores_.append(acc_df)
            kt_scores_.append(kt_df)
        
        pak_scores_ = pd.concat(pak_scores_, axis = 0) #splits by method
        acc_scores_ = pd.concat(acc_scores_, axis = 0)
        kt_scores_ = pd.concat(kt_scores_, axis = 0)

        pak_scores.append(pak_scores_) #array for each random combination of parameters 
        acc_scores.append(acc_scores_)
        kt_scores.append(kt_scores_)

        pd.concat(pak_scores, axis = 0, keys = np.arange(0, trial+1)).to_csv(os.path.join(save_directory, 'pak.csv')) #itermediate save in case things fail :)
        pd.concat(acc_scores, axis = 0, keys = np.arange(0, trial+1)).to_csv(os.path.join(save_directory, 'acc.csv'))
        pd.concat(kt_scores, axis = 0, keys = np.arange(0, trial+1)).to_csv(os.path.join(save_directory, 'kt.csv'))
    return pak_scores, acc_scores, kt_scores

def linear_mask_(metadata):
    # makes step variable monotonically increeasing for the linear trajectory
    metadata_ = metadata.copy()
    mask_root = metadata_['group'] == 'Path1'
    metadata_.loc[mask_root, 'step'] = 100 - metadata_.loc[mask_root, 'step']

    for i in [2,3,4,5,6]:
        mask = metadata_['group'] == 'Path'+str(i)
        metadata_.loc[mask, 'step'] = 100*(i-1) + metadata_.loc[mask, 'step']

    return metadata_

def branch_mask_(metadata):
    # makes step variable monotonically increeasing for the bifurcation or tree trajectories
    metadata_ = metadata.copy()
    mask_root = metadata_['group'] == 'Path1'
    metadata_.loc[mask_root, 'step'] = 100 - metadata_.loc[mask_root, 'step']

    mask_2 = metadata_['group'] == 'Path2'
    metadata_.loc[mask_2, 'step'] = 100 + metadata_.loc[mask_2, 'step']

    mask_34 = np.isin(metadata_['group'], ['Path3','Path4'])
    metadata_.loc[mask_34, 'step'] = 200 + metadata_.loc[mask_34, 'step']

    mask_56 = np.isin(metadata_['group'], ['Path5','Path6'])
    metadata_.loc[mask_56, 'step'] = 300 + metadata_.loc[mask_56, 'step']

    return metadata_

def _sum_to_one(x):
    x = x / np.sum(x)
    x = x.round(3)
    if np.sum(x) != 1:
        x[0] += 1 - np.sum(x)
    x = x.round(3)
    return x

def sim_binom(adata = None,
            dropout_prob = 0,
            random_state = 0):
    """Simulate dropout noise by undersampling from a binomial distribution with the scale rate proportional to gene means
    Parameters
    adata: anndata.AnnData (default = None)
        annotated data object containing single-cell data (dimensions = cells x features)
    dropout_prob: np.float (default = 0)
        dropout probability
    random_state: int (default = 0)
        random seed
    ----------
    Returns
    adata: anndata.AnnData
        annotated data object containing single-cell data corrupted by dropout noise(dimensions = cells x k)
    ----------
    """     
    np.random.seed(random_state)
    X_log = np.log1p(adata.X)
    scale = np.exp(-dropout_prob*(X_log.mean(0))**2)
    adata.X = np.random.binomial(n = adata.X.astype('int'), p = scale, size = adata.shape)
    return adata