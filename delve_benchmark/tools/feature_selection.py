import numpy as np
import scipy
import anndata
import pandas as pd
from scipy.sparse import *
import scanpy as sc
import delve_benchmark
import scmer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import hotspot
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
import logging
logging.basicConfig(level = logging.INFO)

def delve_fs(adata = None,
            k = 10,
            num_subsamples = 1000,
            n_clusters = 5,
            null_iterations = 1000,
            random_state = 0,
            n_random_state = 10,
            n_pcs = None,
            n_jobs = -1,
            return_modules: bool = False,
            **args):
    """Performs DELVE feature selection 
        - step 1: identifies dynamic seed features to construct a between-cell affinity graph according to dynamic cell state progression
        - step 2: ranks features according to their total variation in signal along the approximate trajectory graph using the Laplacian score
    
    Parameters
    adata: anndata.AnnData
        annotated data object where adata.X is the attribute for preprocessed data (dimensions = cells x features)
    k: int (default = 10)
        number of nearest neighbors for between cell affinity kNN graph construction
    num_subsamples: int (default = 1000)
        number of neighborhoods to subsample when estimating feature dynamics  
    n_clusters: int (default = 5)
        number of feature modules
    null_iterations: int (default = 1000)
        number of iterations for gene-wise permutation testing
    random_state: int (default = 0)
        random seed parameter
    n_random_state: int (default = 10)
        number of kmeans clustering initializations
    n_pcs: int (default = None)
        number of principal components to compute pairwise Euclidean distances for between-cell affinity graph construction. If None, uses adata.X
    n_jobs = int (default = -1)
        number of tasks
    ----------
    Returns
    delta_mean: pd.DataFrame
        dataframe containing average pairwise change in expression of all features across subsampled neighborhoods (dimensions = num_subsamples x features)
    modules: pd.DataFrame
        dataframe containing feature-cluster assignment (dimensions = features x 1)
    selected_features: pd.DataFrame
        dataframe containing ranked features and Laplacian scores following feature selection (dimensions = features x 1)
    ----------
    """
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    elif n_jobs < -1:
        n_jobs = mp.cpu_count() + 1 + n_jobs

    X, feature_names, obs_names = delve_benchmark.pp.parse_input(adata) #parse anndata

    try:
        logging.info(f'Step 1: identifying dynamic feature modules')
        sub_idx, _, delta_mean, modules  = seed_select(X = X, feature_names = feature_names, obs_names = obs_names, k = k, num_subsamples = num_subsamples,
                                                    n_clusters = n_clusters, null_iterations = null_iterations, random_state = random_state,
                                                    n_random_state = n_random_state, n_pcs = n_pcs, n_jobs = n_jobs)

        logging.info(f'Step 2: performing feature selection')
        dyn_feats = np.asarray(modules.index[modules['cluster_id'] != 'static'])
        selected_features = feature_select(X = X[sub_idx, :], feature_names = feature_names, dyn_feats = dyn_feats, k = k, n_pcs = n_pcs, n_jobs = n_jobs)

        if return_modules == True:
            return delta_mean, modules, selected_features
        else:
            return selected_features

    except TypeError: #no dynamic seed features were identified
        if return_modules == True:
            return None, None, None
        else:
            return None

def seed_select(X = None,
                feature_names = None,
                obs_names = None,
                k = 10, 
                num_subsamples = 1000,
                n_clusters = 5,
                null_iterations = 1000,
                random_state = 0,
                n_random_state = 10,
                n_pcs = None,
                n_jobs = -1):
    """Identifies dynamic seed clusters
    Parameters
    X: np.ndarray (default = None)
        array containing normalized and preprocessed data (dimensions = cells x features)
    feature_names: np.ndarray (default = None)
        array containing feature names
    obs_names: np.ndarray (default = None)
        array containing cell names   
    k: int (default = 10)
        number of nearest neighbors for between cell affinity kNN graph construction
    num_subsamples: int (default = 1000)
        number of neighborhoods to subsample when estimating feature dynamics  
    n_clusters: int (default = 5)
        number of feature modules
    null_iterations: int (default = 1000)
        number of iterations for gene-wise permutation testing
    random_state: int (default = 0)
        random seed parameter
    n_random_state: int (default = 10)
        number of kmeans clustering initializations
    n_pcs: int (default = None)
        number of principal components to compute pairwise Euclidean distances for between-cell affinity graph construction. If None, uses adata.X
    n_jobs = int (default = -1)
        number of tasks
    ----------
    Returns
    sub_idx: np.andarray
        array containing indices of subsampled neighborhoods
    adata_sub: anndata.AnnData
        annotated data object containing subsampled means (dimensions = num_subsamples x features)
    delta_mean: pd.DataFrame
        dataframe containing average pairwise change in expression of all features across subsampled neighborhoods (dimensions = num_subsamples x features)
    modules: pd.DataFrame
        dataframe containing feature-cluster assignment (dimensions = features x 1)
    ----------
    """                
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    else:
        n_jobs == mp.cpu_count() + 1 + n_jobs

    p = mp.Pool(n_jobs)
    
    np.random.seed(random_state)
    random_state_arr = np.random.randint(0, 1000000, n_random_state)

    logging.info(f'estimating feature dynamics')
    sub_idx, adata_sub, delta_mean = delta_exp(X = X, feature_names = feature_names, obs_names = obs_names, k = k, num_subsamples = num_subsamples, random_state = random_state, n_pcs = n_pcs, n_jobs = n_jobs)

    #identify modules
    mapping_df = pd.DataFrame(index = feature_names)
    dyn_feats = []
    random_state_idx = []
    for result in tqdm(p.imap(partial(_run_cluster, delta_mean, feature_names, n_clusters, null_iterations), random_state_arr), 
                            total = n_random_state, desc = 'clustering features and performing gene-wise permutation testing'):        
        if result is not None:
            mapping_df = pd.concat([mapping_df, result[0]], axis = 1)
            dyn_feats.append(result[1])
            random_state_idx.append(result[2])

    if len(dyn_feats) == 0:
        logging.warning(f'No feature clusters have a dynamic variance greater than null. Consider changing the number of clusters or the subsampling size.')
    else:
        dyn_feats = list(np.unique(list(set.intersection(*map(set,dyn_feats)))))
        if len(dyn_feats) == 0:
            logging.warning(f'No features were considered dynamically-expressed across runs.')
        else:
            modules = _annotate_clusters(mapping_df = mapping_df, dyn_feats = dyn_feats, random_state_idx = random_state_idx[-1])  
            n_dynamic_clusters = len(np.unique(modules['cluster_id'][modules['cluster_id'] != 'static']))
            logging.info(f'identified {n_dynamic_clusters} dynamic cluster(s)')
            return sub_idx, adata_sub, delta_mean, modules

def feature_select(X = None,
                    feature_names = None,
                    dyn_feats = None,
                    k: int  = 10,
                    n_pcs = None, 
                    n_jobs: int  = -1):
    """Ranks features along dynamic seed graph using the Laplacian score: https://papers.nips.cc/paper/2005/file/b5b03f06271f8917685d14cea7c6c50a-Paper.pdf
    Parameters
    X: np.ndarray (default = None)
        array containing normalized and preprocessed data (dimensions = cells x features)
    feature_names: np.ndarray (default = None)
        array containing feature names
    dyn_feats: np.ndarray (default = None)
        array containing features that are dynamically expressed. Can consider replacing this with a set of known regulators.
    k: int (default = 10)
        number of nearest neighbors for between cell affinity kNN graph construction
    n_pcs: int (default = None)
        number of principal components to compute pairwise Euclidean distances for between-cell affinity graph construction. If None, uses adata.X
    n_jobs = int (default = -1)
        number of tasks
    ----------
    Returns
    selected_features: pd.DataFrame
        dataframe containing ranked features and Laplacian scores for feature selection (dimensions = features x 1)
    ----------
    """
    f_idx = np.where(np.isin(feature_names, dyn_feats) == True)[0] #index of feature names to construct seed graph
    W = delve_benchmark.tl.construct_affinity(X = X[:, f_idx], k = k, n_pcs = n_pcs, n_jobs = n_jobs) #constructs graph using dynamic seed features
    scores = laplacian_score(X = X, W = W)
    selected_features = pd.DataFrame(scores, index = feature_names, columns = ['DELVE'])
    selected_features = selected_features.sort_values(by = 'DELVE', ascending = True)

    return selected_features

def delta_exp(X = None,
            feature_names = None, 
            obs_names = None,
            k: int = 10,
            num_subsamples: int = 1000,
            random_state: int = 0,
            n_pcs = None,
            n_jobs: int = -1):
    """Estimates change in expression of features across representative cellular neighborhoods
    Parameters
    X: np.ndarray (default = None)
        array containing normalized and preprocessed data (dimensions = cells x features)
    feature_names: np.ndarray (default = None)
        array containing feature names
    obs_names: np.ndarray (default = None)
        array containing cell names   
    k: int (default = 10)
        number of nearest neighbors for between cell affinity kNN graph construction
    num_subsamples: int (default = 1000)
        number of neighborhoods to subsample when estimating feature dynamics  
    random_state: int (default = 0)
        random seed parameter
    n_pcs: int (default = None)
        number of principal components for between-cell affinity graph computation. if None, uses adata.X to find pairwise Euclidean distances 
    n_jobs = int (default = -1)
        number of tasks
    ----------
    Returns
    sub_idx: np.ndarray
        array containing indices of subsampled neighborhoods
    adata_sub: anndata.AnnData
        annotated data object containing subsampled means (dimensions = num_subsamples x features)
    delta_mean: pd.DataFrame (dimensions = num_subsamples x features)
        array containing average pairwise change in expression of all features across subsampled neighborhoods (dimensions = num_subsamples x features)
    ----------
    """
    #construct between cell affinity kNN graph according to all profiled features
    W = delve_benchmark.tl.construct_affinity(X = X, k = k, n_pcs = n_pcs, n_jobs = -1)

    #compute neighborhood means
    n_bool = W.astype(bool)
    n_mean = (X.transpose() @ n_bool) / np.asarray(n_bool.sum(1)).reshape(1,-1)
    n_mean = pd.DataFrame(n_mean.transpose(), index = obs_names, columns = feature_names)

    #perform subsampling of means to get representative neighborhoods using kernel herding sketching: https://dl.acm.org/doi/abs/10.1145/3535508.3545539, https://github.com/CompCy-lab/SketchKH
    sub_idx, adata_sub = delve_benchmark.tl.sketch(anndata.AnnData(n_mean), num_subsamples = num_subsamples, frequency_seed = random_state, n_jobs = n_jobs)

    #compute the average pairwise change in the expression across all neighborhoods for all features
    subsampled_means = np.asarray(adata_sub.X)
    delta_mean = subsampled_means.reshape(-1, 1, subsampled_means.shape[1]) - subsampled_means.reshape(1, -1,subsampled_means.shape[1])
    delta_mean = delta_mean.sum(axis = 1) * (1 / (subsampled_means.shape[0] - 1))
    delta_mean = pd.DataFrame(delta_mean[np.argsort(adata_sub.obs.index)], index = adata_sub.obs.index[np.argsort(adata_sub.obs.index)], columns = adata_sub.var_names) #resort according to subsampled indices

    return sub_idx[0], adata_sub, delta_mean

def _run_cluster(delta_mean, feature_names, n_clusters, null_iterations, state):
    """Multiprocessing function for identifying feature modules and performing gene-wise permutation testing
    Parameters
    delta_mean: pd.DataFrame
        dataframe containing average pairwise change in expression of all features across subsampled neighborhoods (dimensions = num_subsamples x features)
    feature_names: np.ndarray (default = None)
        array containing feature names
    n_clusters: int (default = 5)
        number of feature modules
    null_iterations: int (default = 1000)
        number of iterations for gene-wise permutation testing
    state: int (default = 0)
        random seed parameter
    ----------
    Returns
    mapping_df: pd.DataFrame
        dataframe containing feature to cluster assignments
    dyn_feats: np.ndarray
        array containing features identified as dynamically-expressed following permutation testing
    delta_mean: np.ndarray
        array containing average pairwise change in expression of all features across subsampled neighborhoods (dimensions = num_subsamples x features)
    state: int
        random seed parameter
    ----------
    """     
    #perform clustering     
    clusters = KMeans(n_clusters = n_clusters, random_state = state).fit_predict(delta_mean.transpose())
    feats = {i:feature_names[np.where(clusters == i)[0]] for i in np.unique(clusters)}

    #record feature-cluster assignment to find intersection across runs
    mapping = np.full((len(feature_names), 1), 'NaN')
    for id, feature in feats.items():
        mapping[np.isin(feature_names, feature)] = str(id)  
    mapping_df = pd.DataFrame(mapping, index = feature_names, columns = [state])

    #compute variance-based permutation test
    seed_var = np.array([np.var(delta_mean.iloc[:, np.isin(feature_names, feats[i])], axis = 1, ddof = 1).mean() for i in range(n_clusters)])
    null_var = []
    for f in range(0, len(feats)):
        null_var_ = np.array([np.var(delta_mean.iloc[:, np.isin(feature_names, np.random.choice(feature_names, len(feats[f]), replace = False))], axis = 1, ddof=1).mean() for i in range(null_iterations)])
        null_var.append(np.mean(null_var_))

    dynamic_id = np.where(seed_var > np.array(null_var))[0] #select dynamic clusters over null variance threshold

    if len(dynamic_id) != 0:
        dyn_feats = np.concatenate([v for k, v in feats.items() if k in np.array(list(feats.keys()))[dynamic_id]])
        return mapping_df, dyn_feats, state

def _annotate_clusters(mapping_df = None,
                        dyn_feats = None,
                        random_state_idx: int = None):
    """Annotates clusters as dynamic or static according to gene-wise permutation testing within clusters
    Parameters
    mapping_df: pd.DataFrame
        dataframe containing feature-cluster ids from KMeans clustering across random trials (dimensions = features x n_random_state)
    dyn_feats:  np.ndarray
        array containing features considered to be dynamically expressed across runs
    random_state_idx:  int (default = None)
        id of random state column id in mapping DataFrame to obtain cluster ids  
    ----------
    Returns
    modules: pd.DataFrame
        dataframe containing annotated feature-cluster assignment (dimensions = features x 1)
    ----------
    """
    cluster_id = np.unique(mapping_df.values)
    dynamic_id = np.unique(mapping_df.loc[dyn_feats].loc[:, random_state_idx])
    static_id = cluster_id[~np.isin(cluster_id, dynamic_id)]

    cats = {id_: 'static' for id_ in static_id}
    cats.update({id_: f'dynamic {i}' if len(dynamic_id) > 1 else 'dynamic' for i, id_ in enumerate(dynamic_id)})

    modules = pd.Categorical(pd.Series(mapping_df.loc[:, random_state_idx].astype('str')).map(cats))
    modules = pd.DataFrame(modules, index = mapping_df.index, columns = ['cluster_id'])

    return modules

def scmer_fs(adata = None,
            k: int  = 10,
            n_pcs: int = 50,
            perplexity: int = 100,
            n_jobs: int  = -1,
            random_state: int = 0,
            **args):
    """Performs feature selection using SCMER: https://www.nature.com/articles/s43588-021-00070-7
    Parameters
    adata: anndata.AnnData (default = None)
        annotated data object containing preprocessed data (dimensions = cells x features)
    k: int (default = 10)
        integer referring to the number of nearest neighbors when computing KNN graph 
    n_pcs: int (default = 50)
        integer referring to the number of principal components
    perplexity: int = 100
        integer referring to the perplexity parameter in SCMER
    n_jobs: int (default = -1)
        integer referring to the number of jobs for parallelization
    random_state: int (default = 0)
        integer referring to the random seed for reproducibility
    ----------

    Returns
    predicted_features: pd.core.frame.DataFrame
        dataframe containing the features following selection and their associated weight (dimensions = features x 1)
    ----------
    """
    sc.tl.pca(adata, n_comps = n_pcs, svd_solver = 'arpack')
    sc.pp.neighbors(adata, knn = k)
    sc.tl.umap(adata)

    model = scmer.UmapL1(lasso = 3.87e-4, ridge = 0., n_pcs = n_pcs, perplexity = perplexity, use_beta_in_Q = True, n_threads = n_jobs, pca_seed = random_state)
    model.fit(adata.X)

    predicted_features = pd.DataFrame({'SCMER': model.w[model.w > 0]}, adata.var_names[model.get_mask()])
    predicted_features = predicted_features.sort_values(by = 'SCMER', ascending=False)

    return predicted_features

def hotspot_fs(adata = None,
            k: int  = 10,
            n_pcs: int = 50,
            **args):
    """Ranks features using hotspot: https://www.sciencedirect.com/science/article/pii/S2405471221001149
    Parameters
    adata: anndata.AnnData (default = None)
        annotated data object containing preprocessed single-cell data (dimensions = cells x features)
    k: int (default = 10)
        number of nearest neighbors for between cell affinity kNN graph construction
    n_pcs: int (default = None)
        number of principal components to compute pairwise Euclidean distances for between-cell affinity graph construction. If None, uses adata.X
    ----------
    Returns
    predicted_features: pd.DataFrame
        dataframe containing ranked features following feature selection (dimensions = features x 1)
    -------
    """
    X, _, _ = delve_benchmark.pp.parse_input(adata)
    adata.layers['data'] = X.copy()
    sc.tl.pca(adata, n_comps = n_pcs, svd_solver = 'arpack')

    hs = hotspot.Hotspot(adata,
                        layer_key="data",
                        model='danb',
                        latent_obsm_key='X_pca')

    hs.create_knn_graph(weighted_graph = True, n_neighbors = k)
    hs_results = hs.compute_autocorrelations()
    hs_genes = hs_results.loc[hs_results.FDR < 0.05]['FDR']
    predicted_features = pd.DataFrame(hs_genes)
    predicted_features.columns = ['hotspot']
    predicted_features.sort_values(by = 'hotspot', ascending=True)

    return predicted_features

def laplacian_score_fs(adata = None,
                    k: int  = None,
                    n_pcs: int = None,
                    n_jobs: int  = -1,
                    **args):
    """Ranks features using the Laplacian score: https://papers.nips.cc/paper/2005/file/b5b03f06271f8917685d14cea7c6c50a-Paper.pdf
    Parameters
    adata: anndata.AnnData (default = None)
        annotated data object containing preprocessed single-cell data (dimensions = cells x features)
    k: int (default = 10)
        number of nearest neighbors for between cell affinity kNN graph construction
    n_pcs: int (default = None)
        number of principal components to compute pairwise Euclidean distances for between-cell affinity graph construction. If None, uses adata.X
    n_jobs = int (default = -1)
        number of tasks
    ----------
    Returns
    predicted_features: pd.DataFrame
        dataframe containing ranked features following feature selection (dimensions = features x 1)
    -------
    """
    X, feature_names, _ = delve_benchmark.pp.parse_input(adata)
    W = delve_benchmark.tl.construct_affinity(X = X, k = k, n_pcs = n_pcs, n_jobs = n_jobs)
    scores = delve_benchmark.tl.laplacian_score(X = X, W = W)
    predicted_features = pd.DataFrame(scores, index = feature_names, columns = ['laplacian_score'])
    predicted_features = predicted_features.sort_values(by = 'laplacian_score', ascending = True)

    return predicted_features   


def laplacian_score(X = None,
                    W = None):
    """Computes the Laplacian score: https://papers.nips.cc/paper/2005/file/b5b03f06271f8917685d14cea7c6c50a-Paper.pdf
    Parameters
    X: np.ndarray (default = None)
        array containing normalized and preprocessed data (dimensions = cells x features)
    W: np.ndarray (default = None)
        adjacency matrix containing between-cell affinity weights
    ----------
    Returns
    l_score: np.ndarray
        array containing laplacian score for all features (dimensions = features)
    ----------
    """
    n_samples, n_features = X.shape
    
    #compute degree matrix
    D = np.array(W.sum(axis = 1))
    D = scipy.sparse.diags(np.transpose(D), [0])

    #compute graph laplacian
    L = D - W.toarray()

    #ones vector: 1 = [1,···,1]'
    ones = np.ones((n_samples,n_features))

    #feature vector: fr = [fr1,...,frm]'
    fr = X.copy()

    #construct fr_t = fr - (fr' D 1/ 1' D 1) 1
    numerator = np.matmul(np.matmul(np.transpose(fr), D.toarray()), ones)
    denomerator = np.matmul(np.matmul(np.transpose(ones), D.toarray()), ones)
    ratio = numerator / denomerator
    ratio = ratio[:, 0]
    ratio = np.tile(ratio, (n_samples, 1))
    fr_t = fr - ratio

    #compute laplacian score Lr = fr_t' L fr_t / fr_t' D fr_t
    l_score = np.matmul(np.matmul(np.transpose(fr_t), L), fr_t) / np.matmul(np.dot(np.transpose(fr_t), D.toarray()), fr_t)
    l_score = np.diag(l_score)

    return l_score

def mcfs_fs(adata = None,
            k: int  = None,
            n_pcs: int = None, 
            n_clusters: int = None,
            n_selected_features = None,
            n_jobs: int  = -1,
            **args):
    """Performs feature selection using multi-cluster feature selection (MCFS): http://people.cs.uchicago.edu/~xiaofei/SIGKDD2010-Cai.pdf

    Parameters
    adata: anndata.AnnData (default = None)
        annotated data object containing preprocessed single-cell data (dimensions = cells x features)
    k: int (default = 10)
        integer referring to the number of nearest neighbors when computing KNN graph 
    n_pcs: int (default = None)
        number of principal components to compute pairwise Euclidean distances for between-cell affinity graph construction. If None, uses adata.X
    n_clusters: int or str (default = None)
        number of eigenvectors to retain following eigendecomposition
    n_selected_features: int (default = None)
        number of nonzero coefficients, of features to select
    n_jobs: int (default = -1)
        integer referring to the number of jobs for parallelization
    ----------

    Returns
    predicted_features: pd.core.frame.DataFrame
        DataFrame containing the MCFS score for selected features (dimensions = n_selected_features x 1). This is sorted (max: best to min)
    ----------
    """
    X, feature_names, _ = delve_benchmark.pp.parse_input(adata)
    W = delve_benchmark.tl.construct_affinity(X = X, k = k, n_pcs = n_pcs, n_jobs = n_jobs)
    predicted_features = mcfs(X = X, W = W, n_selected_features = n_selected_features, n_clusters = n_clusters, feature_names = feature_names)

    return predicted_features

def mcfs(X = None,
        W = None,
        n_selected_features: int = None,
        n_clusters: int = None,
        feature_names = None,
        **kwargs):
    """computes multi-cluster feature selection (MCFS) score: http://people.cs.uchicago.edu/~xiaofei/SIGKDD2010-Cai.pdf

    Parameters
    X: np.ndarray (default = None)
        preprocessed single-cell data (dimensions = cells x features)
    W: np.ndarray (default = None)
        between-cell pairwise affinity matrix
    n_selected_features: int (default = None)
        number of nonzero coefficients, or features to select
    n_clusters: int (default = None)
        number of eigenvectors to retain following eigendecomposition
    feature_names: list (default = None)
        list containing feature names
    ----------

    Returns
    predicted_features: pd.core.frame.DataFrame
        dataframe containing the MCFS score for selected features (dimensions = n_selected_features x 1). This is sorted (max: best to min)
    ----------
    """
    import warnings
    warnings.filterwarnings("ignore")

    #perform eigendecomposition and return top K eigenvectors with respect to smallest eigenvalues
    W = W.toarray()
    W = (W + W.transpose()) / 2 #make symmetric
    W_norm = np.diag(np.sqrt(1 / W.sum(1)))
    W = np.dot(W_norm, np.dot(W, W_norm))
    W_t = W.transpose()
    W[W < W_t] = W_t[W < W_t]

    try:
        _, evecs = scipy.linalg.eigh(a = W) 
        Y = np.dot(W_norm, evecs[:, -1*n_clusters-1:-1])

        #L1-regularized regression using LARs algorithm with cardinality constraint being d
        n_features = X.shape[1]
        W = np.zeros((n_features, n_clusters))

        for i in range(n_clusters):
            clf = linear_model.Lars(n_nonzero_coefs = n_selected_features)
            clf.fit(X, Y[:, i])
            W[:, i] = clf.coef_ #return coefficients for each eigenvector

        mcfs_score = W.max(axis = 1) #take the maximum for MCFS score 
        idx = np.argsort(mcfs_score, 0)
        idx = idx[::-1]

        predicted_features = pd.DataFrame(mcfs_score[idx], index = feature_names[idx], columns = ['MCFS'])
        predicted_features = predicted_features[:n_selected_features]
        return predicted_features
    except:
        return None

def neighborhood_variance_fs(adata = None,
                            **args):
    """Performs feature selection using neighborhood variance: https://genomebiology.biomedcentral.com/articles/10.1186/s13059-016-0975-3

    Parameters
    adata: anndata.AnnData (default = None)
        Anndata object containing single-cell data
    ----------

    Returns
    predicted_features: pd.core.frame.DataFrame
        dataframe containing the neighborhood variance score for selected features
    ----------
    """
    X, feature_names, _ = delve_benchmark.pp.parse_input(adata)
    nvr, nvr_idx = neighborhood_variance_main(X = X, feature_names = feature_names)

    predicted_features = pd.DataFrame(nvr[nvr_idx], index = feature_names[nvr_idx], columns = ['neighborhood_variance'])
    predicted_features = predicted_features.sort_values('neighborhood_variance', ascending = False)

    return predicted_features

def neighborhood_variance_main(X = None):
    """Performs neighborhood variance feature selection

    Parameters
    X: np.ndarray (default = None)
        preprocessed single-cell data (dimensions = cells x features)
    ----------

    Returns
    nvr: np.ndarray
        array containing neighborhood variance scores
    nvr_idx: np.ndarray
        array containing the indicies of features
    -----         
    """     
    sample_variance = np.var(X, axis = 0) #sample variance of features across all cells dimensions = nfeatures
    A, g = min_conn_knn(X = X)
    nv = neighborhood_variance(X = X, A = A, g = g)

    nvr = np.divide(sample_variance, nv)
    nvr_idx = np.nan_to_num(nvr > 1.0)
    return nvr, nvr_idx

def min_conn_knn(X = None):
    """Finds the minimum number of nearest neighbors such that there is a fully connected graph with 1 component for neighborhood variance feature selection

    Parameters
    X: np.ndarray (default = None)
        preprocessed single-cell data (dimensions = cells x features)
    ----------

    Returns
    A: np.ndarray (default = None)
        adjacency matrix of distances between cells
    g: igraph object
        minimally connected graph with dimensions (nsamples x nsamples)
    ----------
    """
    connected_components = 0
    k = 0
    while connected_components != 1:
        k += 1
        knn_tree = NearestNeighbors(n_neighbors = k, n_jobs=-1).fit(X)
        A = knn_tree.kneighbors_graph(X, mode = 'distance')
        g = delve_benchmark.tl.get_igraph(A.todense(), directed = False) #converts adjacency matrix to symmetric undirected igraph object
        connected_components = len(list(g.components())) #determines if graph is fully connected or not

    print('constructed a minimally connected kNN graph with', k , 'neighbors')

    return A, g

def neighborhood_variance(X = None,
                        A = None,
                        g = None):
    """computes neighborhood variance score for all features

    Parameters
    X: np.ndarray (default = None)
        preprocessed data (dimensions = cells x features)
    A: np.ndarray (default = None)
        adjacency matrix of distances between cells
    g: igraph object
        minimally connected graph with dimensions (nsamples x nsamples) 
    ----------

    Returns
    nv: np.ndarray
        neighborhood variance score for all features
    ----------
    """
    # sum_i^n sum_j^kc (eig - eN(i,j)g)^2
    knn_MSE = np.zeros_like(X)
    nodes = g.vcount()
    kc = A.astype(bool)[0].sum()

    for i in range(nodes):
        neighborhood_i = X[g.neighborhood(i, mode = 'out')]
        knn_MSE[i] = np.sum(np.square((neighborhood_i[0] - neighborhood_i[1:])), axis = 0) ## sum(eig - eN)**2

    summed_MSE = np.sum(knn_MSE, axis = 0)
    nv = (1 / (nodes*kc - 1)) * summed_MSE

    return nv

def variance_score(adata = None,
                    **args):
    """Ranks features using max variance

    Parameters
    adata: anndata.AnnData (default = None)
        annotated data object where adata.X is the attribute for preprocessed data (dimensions = cells x features)
    ----------

    Returns
    predicted_features: pd.core.frame.DataFrame
        dataframe containing the variance score for selected features (dimensions = features x 1)
    ----------
    """
    X, feature_names, _ = delve_benchmark.pp.parse_input(adata)

    mu = np.mean(X, axis = 0)
    var_score = (1 / len(X)) * np.sum((X - mu)**2, axis = 0)

    predicted_features = pd.DataFrame(var_score, index = feature_names, columns = ['variance_score'])
    predicted_features = predicted_features.sort_values(by = 'variance_score', ascending = False)

    return predicted_features

def hvg(adata = None,
        n_top_genes: int = None,
        log: bool = True,
        **args):
    """Performs highly variable gene selection using Scanpy: https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.highly_variable_genes.html

    Parameters
    adata: anndata.AnnData (default = None)
        annotated data object where adata.X is the attribute for preprocessed data (dimensions = cells x features)
    n_top_genes: int (default = None)
        integer referring to the number of highly variable genes to keep
    log: bool (default = True)
        if True, will perform log transformation prior to selection 
    ----------

    Returns
    predicted_features: pd.core.frame.DataFrame
        dataframe containing highly variable genes (dimensions = features x 1)
    ----------
    """
    adata_run_ = adata.copy()
    if log == True:
        sc.pp.log1p(adata_run_)
    sc.pp.highly_variable_genes(adata_run_, flavor = 'seurat', n_top_genes = n_top_genes, inplace = True, subset = True)

    predicted_features = pd.DataFrame(adata_run_.var['dispersions_norm'])
    predicted_features = predicted_features.sort_values(by = ['dispersions_norm'], ascending = False)

    return predicted_features

def random_forest(adata = None,
                X = None,
                feature_names = None,
                labels: list = None,
                labels_key: str = None,
                random_state: int = 0,
                n_splits: int = 10,
                n_jobs: int = -1, 
                return_mean: bool = True,
                **args):
    """Uses random forest for predicting features

    Parameters
    adata: anndata.AnnData (default = None)
        annotated data object containing preprocessed single-cell data (dimensions = cells x features)
    X: np.ndarray (default = None)
        preprocessed single-cell data (dimensions = cells x features). If None, will access from adata.X
    feature_names: list (default = None)
        list containing feature names. If None, will access from adata.var_names
    fs_method: delve_benchmark.tl.function (default = None)
        function specifying the type of feature selection to perform
    fs_method_params: dict (default = None)
        dictionary specifying the input parameters to the feature selection method
    labels: list (default = None)
        list containing ground truth cell type labels for classification
    labels_key: str (default = None)
        string referring to the key in adata.obs with labels. This will be used if labels is unspecified
    random_state: int (default = 0)
        integer of random state for reproducibility
    n_splits: int (default = 10)
        number of random splits for stratified kfold cross validation
    n_jobs: int (default = -1)
        integer referring to the number of jobs for parallelization
    return_mean: bool (default = True)
        boolean referring to whether the mean of feature importance scores across folds should be computed
            if True: predicted features are returned
            if False: predicted features and scores are returned
    ----------

    Returns
    feature_importances_df: pd.core.frame.DataFrame
        dataframe containing the predicted features according to feature importance scores
    scores: pd.core.frame.DataFrame
        dataframe containing the scores for classification
    ----------
    """
    if X is None:
        X = adata.X.copy()
        if scipy.sparse.issparse(X):
            X = X.todense()
        feature_names = np.asarray(adata.var_names)
    if labels is None:
        labels = np.asarray(adata.obs[labels_key].values)

    le = LabelEncoder()
    y = le.fit_transform(labels).astype(int)

    feature_importances_df = pd.DataFrame()
    scores = []
    cv = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = random_state)
    for train_ix, test_ix in cv.split(X, y):
        X_train, X_test = X[train_ix, :], X[test_ix, :]
        y_train, y_test = y[train_ix], y[test_ix]

        cv_hyperparam = StratifiedKFold(n_splits = 3, shuffle = True, random_state = random_state)
        model = RandomForestClassifier(random_state = random_state, n_jobs = n_jobs)
        param_grid = {'n_estimators': [10, 100, 500]}
        gsearch = GridSearchCV(model, param_grid, scoring = 'accuracy', n_jobs = n_jobs, cv = cv_hyperparam, refit = True)
        result = gsearch.fit(X_train, y_train) #execute gridsearch
        best_model = result.best_estimator_ #access best performing params
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_true = y_test, y_pred = y_pred)
        scores.append(acc)

        #access feature importance scores and sort
        f_sorted_idx = np.argsort(best_model.feature_importances_)[::-1]
        f_scores = best_model.feature_importances_[f_sorted_idx]
        f_sort_labels = [feature_names[i] for i in f_sorted_idx]

        feature_importances_df_ = pd.DataFrame(f_scores, index = f_sort_labels)
        feature_importances_df = pd.concat([feature_importances_df, feature_importances_df_], axis = 1)
        
    scores = pd.DataFrame(np.asarray(scores), index = np.shape(scores)[0]*['accuracy'])

    if return_mean == True:
        predicted_features = pd.DataFrame(feature_importances_df.mean(axis = 1), columns = ['random_forest'])
        predicted_features = predicted_features.sort_values(by = ['random_forest'], ascending = False)
        return predicted_features
    else:
        return scores, feature_importances_df

def seed_features(adata = None, 
                k: int = 10, 
                num_subsamples: int = 1000,
                n_clusters: int = 5,
                null_iterations: int = 1000,
                random_state: int = 0,
                n_random_state: int = 10,
                n_pcs: int = None,
                n_jobs = -1, 
                **args):
    """Performs step 1 of DELVE feature selection
    Parameters
    adata: anndata.AnnData (default = None)
        annotated data object containing single-cell data (dimensions: cells x features)     
    k: int (default = 10)
        number of nearest neighbors for between cell affinity kNN graph construction
    num_subsamples: int (default = 1000)
        number of neighborhoods to subsample when estimating feature dynamics  
    n_clusters: int (default = 5)
        number of feature modules
    null_iterations: int (default = 1000)
        number of iterations for gene-wise permutation testing
    random_state: int (default = 0)
        random seed parameter
    n_random_state: int (default = 10)
        number of kmeans clustering initializations
    n_pcs: int (default = None)
        number of principal components to compute pairwise Euclidean distances for between-cell affinity graph construction. If None, uses adata.X
    n_jobs = int (default = -1)
        number of tasks
    ----------
    Returns
    predicted_features: pd.core.frame.DataFrame
        dataframe containing dynamic seed features (dimensions = features x 1)
    -----
    """
    X, feature_names, obs_names = delve_benchmark.pp.parse_input(adata)

    try:
        _, _, _, modules  = seed_select(X = X, feature_names = feature_names, obs_names = obs_names, k = k, num_subsamples = num_subsamples,
                                        n_clusters = n_clusters, null_iterations = null_iterations, random_state = random_state,
                                        n_random_state = n_random_state, n_pcs = n_pcs, n_jobs = n_jobs)

        seed_features = np.concatenate([modules.index[modules['cluster_id'].values == i] for i in [i for i in np.unique(modules['cluster_id']) if 'dynamic' in i]])
    except:
        seed_features = None

    return seed_features

def all_features(adata = None,
                **args):
    """Returns all features

    Parameters
    adata: anndata.AnnData (default = None)
        annotated data object containing single-cell data (dimensions: cells x features)
    ----------

    Returns
    feature_names: np.ndarray
        array containing all features
    ----------
    """
    _, feature_names, _ = delve_benchmark.pp.parse_input(adata)

    return feature_names

def random_features(adata = None,
                random_state: int = 0,
                **args):
    """Returns random set of features

    Parameters
    adata: anndata.AnnData (default = None)
        Anndata object containing single-cell data
    random_state: int (default = 0)
        integer referring to the random seed =
    ----------

    Returns
    rand_features: np.ndarray
        array containing all features
    ----------
    """
    _, feature_names, _ = delve_benchmark.pp.parse_input(adata)

    np.random.seed(random_state)
    rand_features = np.random.choice(feature_names, size = len(feature_names), replace = False)
    
    return rand_features