import numpy as np
import pandas as pd
import scipy
import networkx as nx
import scanpy as sc
from scipy.sparse import *
from scipy import stats
from scipy.stats import kendalltau
import anndata
import statsmodels
import statsmodels.api as sm 
from statsmodels.gam.api import GLMGam, BSplines, CyclicCubicSplines
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, accuracy_score, mean_squared_error
from sklearn.model_selection import StratifiedKFold, GridSearchCV, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC, SVR
import gseapy as gp
import requests
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
import phate
import os
import delve_benchmark
pandas2ri.activate()

def pak(predicted_features = None,
        reference_features = None,
        k_sweep = None, 
        **args):
    """Computes precision at k (p@k) score
    Parameters
    predicted_features: np.ndarray (default = None)
        array containing predicted features from a feature selection method
    reference_features: np.ndarray (default = None)
        array containing reference features for comparison
    k_sweep: list (default = None)
        list containing k values for precision scores 
    ----------
    Returns
    scores: pd.core.Frame.DataFrame
        dataframe containing p@k scores
    ----------
    """
    predicted_set = set(predicted_features)
    reference_set = set(reference_features)
    overlap = len(predicted_set.intersection(reference_set))
    n_predicted = len(predicted_set)
    
    if n_predicted < min(k_sweep):
        pak_arr = overlap / n_predicted
        scores = pd.DataFrame([pak_arr], index=[n_predicted])
    else:
        max_k = min(max(k_sweep), n_predicted)
        k_sweep = np.array(k_sweep)
        k_sweep = k_sweep[k_sweep <= max_k]
        pak_arr = np.zeros_like(k_sweep, dtype = float)
        for i, k in enumerate(k_sweep):
            k_predicted_set = predicted_set.copy()
            k_predicted = predicted_features[:k]
            k_predicted_set.intersection_update(set(k_predicted))
            overlap_k = len(k_predicted_set.intersection(reference_set))
            total_k = len(k_predicted_set)
            pak_arr[i] = overlap_k / total_k
        scores = pd.DataFrame(pak_arr, index = k_sweep)

    return scores

def pak_all(predicted_features = None,
            reference_features = None,
            k_sweep = None, 
            **args):
    """Computes precision at k score (p@k) for the total feature set (all features)
    Parameters
    predicted_features: np.ndarray (default = None)
        array containing all features
    reference_features: np.ndarray (default = None)
        array containing reference features for comparison
    k_sweep: list (default = None)
        list containing k values for precision scores 
    ----------
    Returns
    scores: pd.core.Frame.DataFrame
        dataframe containing p@k scores
    ----------
    """
    predicted_set = set(predicted_features)
    reference_set = set(reference_features)
    overlap = len(predicted_set.intersection(reference_set))
    n_predicted = len(predicted_set)
    pak_arr = overlap / n_predicted

    scores = pd.DataFrame([pak_arr]*len(k_sweep), index = k_sweep)
    
    return scores

def cluster(adata = None,
            X = None,
            feature_names = None,
            labels = None,
            labels_key: str = None,
            n_clusters: int = 4,
            n_sweep: int = 25,
            predicted_features = None,
            **args):
    """Performs KMeans clustering and then computes NMI between clusters from predicted feature set and ground truth cell types
    Parameters
        adata: anndata.AnnData (default = None)
            annotated data object containing single-cell data (dimensions = cells x features)
        X: np.ndarray (default = None)
            data (dimensions = cells x features). If None, will access from adata.X
        feature_names: list (default = None)
            list containing feature names. If None, will access from adata.var_names
        predicted_features: np.ndarray (default = None)
            array containing predicted features from a feature selection method
        labels: list (default = None)
            list containing cell type labels
        labels_key: str (default = None)
            string referring to the key in adata object with ground truth labels. This will be used if 'labels' is unspecified
        n_clusters: int (default = 5)
            integer referring to the number of clusters in KMeans
        n_sweep: int (default = 25)
            integer referring to how many random states to sweep through
    ----------
    Returns
        scores: pd.core.Frame.DataFrame
            dataframe containing NMI clustering scores
    ----------
    """
    if X is None:
        X = adata.X.copy()
        if scipy.sparse.issparse(X):
            X = X.todense()
        feature_names = np.asarray(adata.var_names)
    if labels_key is not None:
        labels = np.asarray(adata.obs[labels_key].values)

    sweep = np.random.randint(0, 10000, n_sweep)
    scores = []
    for i in sweep:
        y_pred = KMeans(n_clusters = n_clusters, random_state = i).fit(X[:, np.isin(feature_names, predicted_features)]).labels_.astype('str')
        nmi = normalized_mutual_info_score(labels, y_pred)
        scores.append(nmi)

    scores = pd.DataFrame(scores)
    scores.index = len(scores)*['NMI']

    return scores

def perform_dpt(adata = None,
                k: int = 10,
                n_dcs: int = 20,
                n_pcs: int = 50, 
                root: int = None,
                **args):
    """Performs trajectory inference using diffusion pseudotime: https://www.nature.com/articles/nmeth.3971
    Parameters
        adata: anndata.AnnData (default = None)
            annotated data object containing single-cell data (dimensions = cells x features)
        k: int (default = 10)
            number of nearest neighbors for kNN graph construction
        n_pcs: int (default = 50)
            number of principal components for computing pairwise Euclidean distances in kNN graph construction. If 0, will use adata.X
        n_dcs: int (default = 20)
            integer referring to the number of diffusion map components
        root: int (default = None)
            integer referring to root cell to impose directionality
    ----------
    Returns
        adata: anndata.AnnData (default = None)
            annotated data object containing single-cell data (dimensions = cells x features)
        pseudotime: np.ndarray
            array containing pseudotime values for all cells (dimensions = cells x 1)
    ----------
    """  
    sc.pp.neighbors(adata, n_neighbors = k, n_pcs = n_pcs)
    sc.tl.diffmap(adata, n_comps = n_dcs)
    adata.uns['iroot'] = root
    sc.tl.dpt(adata, n_dcs = n_dcs)
    pseudotime = np.asarray(adata.obs['dpt_pseudotime'].values).flatten()
    return adata, pseudotime

def perform_slingshot(adata = None,
                        cluster_labels_key: str = None,
                        root_cluster: str = None,
                        k = 30,
                        t = 10,
                        random_state = 0,
                        **args):
    """performs trajectory inference using slingshot: https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-018-4772-0
​
    Parameters
    adata: AnnData
        annotated data object
    cluster_labels_key: str (default = None)
        string referring to the cell population labels for inference
    root_cluster: str (default = None)
        string referring to the starting root population
    k: int (default = 30): 
        number of nearest neighbors
    t: int (default = 10)
        power of diffusion operator in PHATE embedding computation
    random_state: int (default = 0)
        random seed
    ----------
​
    Returns
​    adata: anndata.AnnData (default = None)
        annotated data object containing single-cell data (dimensions = cells x features)
    avg_pseudotime: np.ndarray
        array containing pseudotime values for all cells averaged over all estimated trajectories (dimensions = cells x 1)
    ----------
    """
    labels = adata.obs[cluster_labels_key].copy()
    op = phate.PHATE(knn = k, t = t, random_state = random_state)
    X_phate = op.fit_transform(adata)
    adata.obsm['X_phate'] = X_phate.copy()
    embedding = pd.DataFrame(X_phate, index = adata.obs_names)
    
    r = robjects.r
    r['source'](os.path.join('delve_benchmark', 'tools', 'evaluate.r'))
    slingshot_r = robjects.globalenv['run_slingshot']
    result_r = slingshot_r(pandas2ri.py2rpy(embedding), pandas2ri.py2rpy(labels), root_cluster)
    lineages = result_r[0]
    pseudotime = result_r[1]
    try:
        avg_pseudotime = np.nanmean([pseudotime[:, i] for i in range(0, np.shape(pseudotime)[1])], axis = 0)
    except:
        avg_pseudotime = [np.nan]*len(adata.obs_names)

    adata.obs['pseduotime'] = avg_pseudotime
    return adata, avg_pseudotime

def compute_de(adata = None,
                de_method = None,
                family: str = 'neg-binomial', 
                de_params: dict = {'k': 10, 'n_pcs': 0, 'n_dcs': 20}):
    """Performs trajectory differential expression analysis
    Parameters
        adata: anndata.AnnData
            annotated data object containing single-cell data (dimensions = cells x features)
        de_method: delve_benchmark function for performing trajectory inference (defualt = None)
            delve_benchmark.tl.perform_dpt or delve_benchmark.tl.perform_slingshot
        family: str (defualt = 'neg_binomial')
            distirbution family
        de_params: dict 
            dictionary containing trajectory inference hyperparaneters 
    ----------
    Returns
        de_df: pd.core.Frame.DataFrame
            dataframe containing features and p-values
    ----------
    """  
    if isinstance(adata.X, scipy.sparse._csr.csr_matrix):
        adata.X = adata.X.toarray()
    
    adata, pseudotime = de_method(adata = adata, **de_params)
    de_dict = de_analy(adata = adata, pse_t = pd.DataFrame(pseudotime, index = adata.obs_names), distri = family)
    de_df = _de_dict2df(de_dict)

    return adata, de_df

def GAM_pt(pse_t, expr, smooth = 'BSplines', df = 5, degree = 3, family = sm.families.NegativeBinomial(alpha = 1.0)):
    """
    ~~ The following code was accessed from CellPath: https://github.com/PeterZZQ/CellPath ~~
    Fit a Generalized Additive Model with the exog to be the pseudo-time. The likelihood ratio test is performed 
    to test the significance of pseudo-time in affecting gene expression value
    Parameters
    ----------
    pse_t
        pseudo-time
    expr
        expression value
    smooth
        choose between BSplines and CyclicCubicSplines
    df
        degree of freedom of the model
    degree
        degree of the spline function
    family
        distribution family to choose, default is negative binomial.
    Returns
    -------
    y_full
        predict regressed value with full model
    y_reduced
        predict regressed value from null hypothesis
    lr_pvalue
        p-value
    """ 
    if smooth == 'BSplines':
        spline = BSplines(pse_t, df = [df], degree = [degree])
    elif smooth == 'CyclicCubicSplines':
        spline = CyclicCubicSplines(pse_t, df = [df])

    exog, endog = sm.add_constant(pse_t),expr
    # calculate full model
    model_full = sm.GLMGam(endog = endog, exog = exog, smoother = spline, family = family)
    try:
        res_full = model_full.fit()
    except:
        return None, None, None
    else:
        # default is exog
        y_full = res_full.predict()
        # reduced model
        y_reduced = res_full.null

        # number of samples - number of paras (res_full.df_resid)
        df_full_residual = expr.shape[0] - df
        df_reduced_residual = expr.shape[0] - 1

        # likelihood of full model
        llf_full = res_full.llf
        # likelihood of reduced(null) model
        llf_reduced = res_full.llnull

        lrdf = (df_reduced_residual - df_full_residual)
        lrstat = -2*(llf_reduced - llf_full)
        lr_pvalue = stats.chi2.sf(lrstat, df=lrdf)
        return y_full, y_reduced, lr_pvalue

def de_analy(adata, pse_t, p_val_t = 0.05, verbose = False, distri = "neg-binomial", fdr_correct = True):
    """
    ~~ The following code was accessed and modified from CellPath: https://github.com/PeterZZQ/CellPath ~~
    Conduct differentially expressed gene analysis.
    Parameters
    ----------
    adata
        annotated data object containing single-cell data (dimensions = cells x features)
    pse_t
        array containing pseudotime values following trajectory inference
    p_val_t
        the threshold of p-value
    verbose
        output the differentially expressed gene
    distri
        distribution of gene expression: either "neg-binomial" or "log-normal"
    fdr_correct
        conduct fdr correction for multiple tests or not
    Returns
    -------
    de_genes
        dictionary that store the differentially expressed genes
    """ 
    pseudo_order = pse_t.copy()

    de_genes = {}
    for reconst_i in pseudo_order.columns:
        de_genes[reconst_i] = []
        sorted_pt = pseudo_order[reconst_i].dropna(axis = 0).sort_values()
        ordering = sorted_pt.index.values.squeeze()

        adata = adata[ordering,:].copy()
        # filter out genes that are expressed in a small proportion of cells 
        sc.pp.filter_genes(adata, min_cells = int(0.05 * ordering.shape[0]))

        for idx, gene in enumerate(adata.var.index):
            gene_dynamic = np.squeeze(adata.X[:,idx])
            pse_t = np.arange(gene_dynamic.shape[0])[:,None]
            if distri == "neg-binomial":
                gene_pred, gene_null, p_val = GAM_pt(pse_t, gene_dynamic, smooth='BSplines', df = 4, degree = 3, family=sm.families.NegativeBinomial(alpha=1.0))
            
            elif distri == "log-normal":                
                gene_pred, gene_null, p_val = GAM_pt(pse_t, gene_dynamic, smooth='BSplines', df = 4, degree = 3, family=sm.families.Gaussian(link = sm.families.links.log()))
            
            else:
                raise ValueError("distribution can only be `neg-binomial` or `log-normal`")

            if p_val != None:
                if verbose:
                    print("gene: ", gene, ", pvalue = ", p_val)
                # if p_val <= p_val_t:
                de_genes[reconst_i].append({"gene": gene, "regression": gene_pred, "null": gene_null,"p_val": p_val})
        
        # sort according to the p_val
        de_genes[reconst_i] = sorted(de_genes[reconst_i], key=lambda x: x["p_val"],reverse=False)

        if fdr_correct:
            pvals = [x["p_val"] for x in de_genes[reconst_i]]
            is_de, pvals = statsmodels.stats.multitest.fdrcorrection(pvals, alpha=p_val_t, method='indep', is_sorted=True)
            
            # update p-value
            for gene_idx in range(len(de_genes[reconst_i])):
                de_genes[reconst_i][gene_idx]["p_val"] = pvals[gene_idx]
            
            # remove the non-de genes
            de_genes[reconst_i] = [x for i,x in enumerate(de_genes[reconst_i]) if is_de[i] == True]
        
    return de_genes

def _de_dict2df(de_dict = None):
    """converts dictionary output to dataframe consisting of features and p-values
    Parameters
        de_dict: dict (default = None)
            dictionary from de_analy
    ----------
    Returns
        de_df: pd.core.Frame.DataFrame
            dataframe containing features and p-values
    ----------
    """ 
    de_genes_df = pd.DataFrame()
    de_pval_df = pd.DataFrame()
    for _, value in de_dict.items():
        for i in range(0, len(value)):
            for key2, value2 in value[i].items():
                if key2 == 'gene':
                    de_genes_df = pd.concat([de_genes_df, pd.DataFrame([value2])], axis = 0)
                elif key2 == 'p_val':
                    de_pval_df = pd.concat([de_pval_df, pd.DataFrame([value2])], axis = 0)

    de_df = de_pval_df.copy()
    de_df.index = de_genes_df[0].values
    de_df.columns = ['pval']
    return de_df

def gene_ontology(gene_list = None,
                  gene_sets = None,
                  organism = 'mouse'):
    """Performs gene set enrichment analysis using Enrichr https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-14-128 and gseapy https://gseapy.readthedocs.io/en/latest/introduction.html
    Parameters
        gene_list: list (default = None)
            list of genes
        gene_sets: list (default = None)
            list of gene sets to consider
        organism: str (default = 'mouse')
            organism for gene ontology
    ----------
    Returns
        go_df: pd.core.Frame.DataFrame
            dataframe containing gene set enrichment analysis results sorted by 'Combined Score'
    ----------
    """  
    enr = gp.enrichr(gene_list = gene_list,
                 gene_sets = gene_sets, 
                 organism = organism, 
                 outdir = None)
    
    go_df = enr.results.sort_values(by = 'Combined Score', ascending = False)
    return go_df

def kt_corr(adata = None,
            X = None,
            feature_names = None,
            k: int = 10,
            n_dcs: int = 20,
            n_pcs: int = 0,
            trajectory_key: list = None,
            trajectories = None,
            roots_arr = None,
            predicted_features = None,
            reference_features = None,
            corr_reference_features: bool = True,
            labels = None,
            labels_key: str = None, 
            **args):
    """Computes Kendall rank correlation between either:
        1. pseudotime array from predicted features and pseudotime array from reference features
        2. pseudotime array from predicted features and ground truth age 
        -- of note, this function is specifically for two distinct trajectories (e.g. proliferation and arrest)
    Parameters
        adata: anndata.AnnData (default = None)
            annotated data object containing single-cell data (dimensions = cells x features)
        X: np.ndarray (default = None)
            data (dimensions = cells x features). If None, will access from adata.X
        feature_names: list (default = None)
            list containing feature names. If None, will access from adata.var_names
        k: int (default = 10)
            number of nearest neighbors for kNN graph construction
        n_pcs: int (default = 0)
            number of principal components for computing pairwise Euclidean distances in kNN graph construction. If 0, will use adata.X
        n_dcs: int (default = 20)
            integer referring to the number of diffusion map components
        root: int (default = None)
            integer referring to root cell to impose directionality
        trajectory_key: str (default = None)
            string specifying where trajectories exist in adata.obs 
        trajectories: np.ndarray (default = None)
            array containing indices of cells for each trajectory
        roots_arr: list (default = None)
            list containing root cells for each trajectory
        predicted_features: np.ndarray (default = None)
            array containing predicted features from a feature selection method
        reference_features: np.ndarray (default = None)
            array containing reference features for comparison
        corr_reference_features: bool (default = True)
            if true: uses ground time labels, else computes pseudotime
        labels: list (default = None)
            list containing cell type labels
        labels_key: str (default = None)
            string referring to the key in adata object with labels. This will be used if labels is unspecified
    ----------
    Returns
        scores: pd.core.Frame.DataFrame
            dataFrame containing Spearman rank correlation scores
    ----------
    """ 
    if adata is None:
        adata = anndata.AnnData(X)
        adata.var_names = feature_names.copy()
    
    if adata is not None:
        feature_names = adata.var_names.copy()

    if trajectories is None:
        classes = np.unique(adata.obs[trajectory_key].values)
        trajectories = []
        for i in range(0, len(classes)):
            trajectories.append(np.where(adata.obs[trajectory_key].values == classes[i])[0])

    adata_objs = []
    for i in range(0, len(trajectories)):
        adata_objs.append(anndata.AnnData(X[trajectories[i], :]))

    if corr_reference_features == True: #compute correlation between two pseudotime arrays
        scores = compute_kt_pseudo(adata_objs = adata_objs, feature_names = feature_names, predicted_features = predicted_features,
                                    reference_features = reference_features, roots_arr = roots_arr, k = k, n_dcs = n_dcs, n_pcs = n_pcs)
    
    else: #compute correlation between pseudotime and ground truth 
        if labels is None:
            labels = adata.obs[labels_key].copy()
        
        y_arr = []
        for i in range(0, len(trajectories)):
            y_arr.append(labels[trajectories[i]])

        scores = compute_kt_labels(adata_objs = adata_objs, feature_names = feature_names, predicted_features = predicted_features,
                                    y_arr = y_arr, roots_arr = roots_arr, k = k, n_dcs = n_dcs, n_pcs = n_pcs)

    return scores

def compute_kt_pseudo(adata_objs = None,
                        feature_names = None,
                        predicted_features = None,
                        reference_features = None,
                        roots_arr = None,
                        k: int = 10,
                        n_dcs: int = 20,
                        n_pcs: int = 0):
    """Computes Kendall rank correlation between pseudotime array from predicted features and pseudotime array from reference features
    Parameters
        adata_objs: anndata.AnnData (default = None)
            annotated data objects containing trajectories
        feature_names: list (default = None)
            list containing feature names
        k: int (default = 10)
            number of nearest neighbors for kNN graph construction
        n_dcs: int (default = 20)
            integer referring to the number of diffusion map components
        n_pcs: int (default = 0)
            number of principal components for computing pairwise Euclidean distances in kNN graph construction. If 0, will use adata.X
        predicted_features: np.ndarray (default = None)
            array containing predicted features from a feature selection method
        reference_features: np.ndarray (default = None)
            array containing reference features for comparison
        roots_arr: list (default = None)
            list containing indices of root cells for each trajectory
    ----------
    Returns
        kt_corr_df: pd.core.Frame.DataFrame
            dataframe containing Kendall rank correlation scores
    ----------
    """ 
    kt_corr_df = pd.DataFrame()
    for i in range(0, len(adata_objs)):
        adata_1 = adata_objs[i][:, np.isin(feature_names, predicted_features)]
        adata_2 = adata_objs[i][:, np.isin(feature_names, reference_features)]
        kt_corr = []
        for root in roots_arr[i]:
            adata_1, pseudotime_1 = perform_dpt(adata = adata_1, k = k, n_dcs = n_dcs, n_pcs = n_pcs, root = root)
            adata_2, pseudotime_2 = perform_dpt(adata = adata_2, k = k, n_dcs = n_dcs,  n_pcs = n_pcs, root = root)
            tau, _ = kendalltau(pseudotime_1, pseudotime_2)
            kt_corr.append(tau)

        kt_corr_df = pd.concat([kt_corr_df, pd.DataFrame(kt_corr, index = len(kt_corr)*['kt_corr_pseudo'])], axis = 1)

    return kt_corr_df

def compute_kt_labels(adata_objs = None,
                        feature_names = None,
                        predicted_features = None,
                        y_arr = None,
                        roots_arr = None,
                        k: int = 10,
                        n_dcs: int = 20,
                        n_pcs: int = 0):
    """Computes Kendall rank correlation between pseudotime array from predicted features and ground truth age
    Parameters
        adata_objs: anndata.AnnData (default = None)
            annotated data objects containing trajectories
        feature_names: list (default = None)
            list containing feature names
        k: int (default = 10)
            number of nearest neighbors for kNN graph construction
        n_dcs: int (default = 20)
            integer referring to the number of diffusion map components
        n_pcs: int (default = 0)
            number of principal components for computing pairwise Euclidean distances in kNN graph construction. If 0, will use adata.X
        predicted_features: np.ndarray (default = None)
            array containing predicted features from a feature selection method
        y_arr: np.ndarray (default = None)
            array containing ground truth age/time labels
        roots_arr: list (default = None)
            list containing indices of root cells for each trajectory
    ----------
    Returns
        kt_corr_df: pd.core.Frame.DataFrame
            dataframe containing Kendall rank correlation scores
    ----------
    """ 
    kt_corr_df = pd.DataFrame()
    for i in range(0, len(adata_objs)):
        adata = adata_objs[i][:, np.isin(feature_names, predicted_features)]
        kt_corr = []
        y_age = y_arr[i]
        for root in roots_arr[i]:
            adata, pseudotime = perform_dpt(adata = adata, k = k, n_dcs = n_dcs, n_pcs = n_pcs, root = root)
            tau, _ = kendalltau(pseudotime, y_age)
            kt_corr.append(tau)

        kt_corr_df = pd.concat([kt_corr_df, pd.DataFrame(kt_corr, index = len(kt_corr)*['kt_corr_labels'])], axis = 1)

    return kt_corr_df

def compute_string_interaction(feats, species = '10090'):
    """Computes protein-protein interaction scores using experimental edges from STRING database: https://string-db.org/
    Parameters
        feats: list of features
        species: code for species
    ----------
    Returns
        interactions: pd.core.Frame.DataFrame
            dataframe containing interaction scores from STRING
    ----------
    """  
    proteins = '%0d'.join(feats)
    url = 'https://string-db.org/api/tsv/network'

    # Parameters for the request
    params = {
        'identifiers': proteins,
        'species': species,
        'network_flavor': 'evidence',
    }
    # url = 'https://string-db.org/api/tsv/network?identifiers=' + proteins + '&species=10090' + '&network_flavor=evidence'
    r = requests.post(url, data = params)

    lines = r.text.split('\n') # pull the text from the response object and split based on new lines
    data = [l.split('\t') for l in lines] # split each line into its components based on tabs
    # convert to dataframe using the first row as the column names; drop empty, final row
    df = pd.DataFrame(data[1:-1], columns = data[0]) 
    # dataframe with the preferred names of the two proteins and the score of the interaction
    interactions = df[['preferredName_A', 'preferredName_B', 'escore']].copy()
    interactions.loc[:, 'escore'] =  pd.DataFrame(interactions.loc[:, 'escore']).astype(float)

    interactions = interactions[np.isin(interactions.loc[:, 'preferredName_A'].values, feats)]
    interactions = interactions[np.isin(interactions.loc[:, 'preferredName_B'].values, feats)]
    return interactions

def compute_string_degree(G):
    """Computes degree score from interaction scores from STRING
    Parameters
        G: networkx degree object
            protein-protein interaction network
    ----------
    Returns
        avg_degree:
            average degree score
    ----------
    """  
    avg_degree = np.mean([v for _, v in G.degree()])

    return avg_degree

def compute_G(interactions):
    """ constructs protein-protein interaction network using STRING association scores
    Parameters
        interactions: pd.core.Frame.DataFrame
            dataframe containing interaction scores from STRING
    ----------
    Returns
        G: networkx degree object
            protein-protein interaction network
    ----------
    """  
    G = nx.Graph(name='Protein Interaction Graph')
    df = interactions[interactions.loc[:, 'escore'] != 0]
    df = np.array(df)
    for i in range(len(df)):
        interaction = df[i]
        a = interaction[0] # protein a node
        b = interaction[1] # protein b node
        w = float(interaction[2]) # score as weighted edge where high scores = low weight
        G.add_weighted_edges_from([(a, b, w)]) # add weighted edge to graph

    return G

def permute_string(modules = None,
                    niterations: int = 1000,
                    colors = ['#B46CDA', '#78CE8B', '#FF8595', '#C9C9C9'],
                    species = '10090', 
                    save_directory = None):
    """Performs permutation test of features within DELVE modules as compared to random assignment using STRING experimental association scores (see figure 5b)
    Parameters
    modules: pd.core.Frame.DataFrame (default = None)
        dataframe containing feature-cluster assignment from DELVE seed selection (step 1)
    niterations: int (default = 1000)
        number of random permutations
    colors: list
        list containing hex codes of colors for each module in the histograms
    species: str 
        species key
    save_directory: str (default = None)
        if specified, will save figure
    ----------
    Returns
    obs_df: pd.core.Frame.DataFrame
        dataframe containing average degree scores for features within each module (observed)
    rand_df: list
        dataframe containing average degree scores for random selection of features per module averaged over all trials
    ----------
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    groups = np.unique(modules['cluster_id'])
    all_feats = np.asarray(modules.index)

    obs_df = []
    rand_df = []
    for g in range(0, len(groups)):
        print(groups[g])
        selected_feats = list(modules.index[modules['cluster_id'] == groups[g]])
        interactions = compute_string_interaction(selected_feats, species = species)
        G = delve_benchmark.tl.compute_G(interactions)
        if len(G.edges()) !=0:
            degree = delve_benchmark.tl.compute_string_degree(G)
        else:
            degree = np.nan

        obs_df.append(pd.DataFrame([degree], index = ['degree']).transpose())
            
        null_degrees =[]
        for i in range(0, niterations):
            print(i)
            rand_feats = list(np.random.choice(all_feats, len(selected_feats), replace=False))
            interactions_null = compute_string_interaction(rand_feats, species = species)
            G_null = delve_benchmark.tl.compute_G(interactions_null)
            if len(G_null.edges()) !=0:
                degree_null = delve_benchmark.tl.compute_string_degree(G_null)
            else:
                degree_null = np.nan
            null_degrees.append(degree_null)

        rand_df.append(pd.DataFrame([null_degrees], index = ['degree']).transpose())

    _, axes = plt.subplots(1, len(colors), figsize = (4.5*len(colors), 3.5), gridspec_kw={'hspace': 0.45, 'wspace': 0.3, 'bottom':0.15})
    sns.set_style('ticks')
    for i, ax in zip(range(0, len(obs_df)), axes.flat):
        obs = obs_df[i]['degree'].values
        permuted = rand_df[i]['degree']

        g = sns.histplot(permuted, ax = ax, color = '#9AA0AC', bins = 10)
        ax.tick_params(labelsize=16)
        g.set_title(f'{groups[i]}', fontsize = 16)
        g.set_xlabel('avg. degree experimental edges', fontsize = 16)
        g.set_ylabel('frequency', fontsize = 16)
        ax.axvline(obs, c = colors[i], lw = 2, ls = '--') #colors[i]

        plt.text(0.55, 0.85, f'pval = {np.round((len(np.where(np.asarray(permuted.values) >= obs)[0]) + 1) / (niterations+1), 3)}', 
            fontsize=14, color=colors[i], #colors[i]
            ha='left', va='bottom',
            transform=ax.transAxes)

    if save_directory is not None:
        delve_benchmark.pp.make_directory(save_directory)
        plt.savefig(os.path.join(save_directory, 'experimental_degree_DELVE.pdf'), bbox_inches = 'tight')

    return obs_df, rand_df

def svm(adata = None,
        X = None,
        feature_names = None,
        fs_method = None,
        fs_method_params = None,
        feature_threshold = None,
        labels: list = None,
        labels_key: str = None,
        random_state: int = 0,
        n_splits: int = 10, 
        n_jobs: int = -1, 
        **args):
    """Performs SVM classification

    Parameters
    adata: anndata.AnnData (default = None)
        annotated data object containing single-cell data (dimensions = cells x features)
    X: np.ndarray (default = None)
        data (dimensions = cells x features). If None, will access from adata.X
    feature_names: list (default = None)
        list containing feature names. If None, will access from adata.var_names
    fs_method: delve_benchmark.tl.function (default = None)
        function specifying the type of feature selection to perform
    fs_method_params: dict (default = None)
        dictionary specifying the input parameters to the feature selection method
    feature_threshold: int (default = None)
        threshold for selecting for features for evaluation. If None, uses all features
    labels: list (default = None)
        list containing ground truth labels for classification
    labels_key: str (default = None)
        string referring to the key in adata.obs containing labels. This will be used if labels is unspecified
    random_state: int (default = 0)
        integer of random state for reproducibility
    n_splits: int (default = 10)
        number of random splits for stratified kfold cross validation
    n_jobs: int (default = -1)
        integer referring to the number of jobs for parallelization
    ----------

    Returns
    predicted_features: np.ndarray
        array of features following feature selection from last fold
    scores: pd.core.frame.DataFrame
        dataframe containing the accuracy scores following classification
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

    scores = []
    cv = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = random_state)
    for train_ix, test_ix in cv.split(X, y):
        X_train, X_test = X[train_ix, :], X[test_ix, :]
        y_train, y_test = y[train_ix], y[test_ix]

        #perform feature selection on training data 
        predicted_features = fs_method(adata = adata[train_ix, :], X = X_train, feature_names = feature_names, n_jobs = n_jobs, **fs_method_params)

        if type(predicted_features) is not pd.core.frame.DataFrame:
            predicted_features = pd.DataFrame(index = predicted_features)        
        if feature_threshold is not None:
            predicted_features = np.asarray(predicted_features.index[:feature_threshold])
        else:
            predicted_features = np.asarray(predicted_features.index)
            
        #subset data according to predicted features
        X_train = X_train[:, np.isin(feature_names, predicted_features)] 
        X_test = X_test[:, np.isin(feature_names, predicted_features)]

        cv_hyperparam = StratifiedKFold(n_splits = 3, shuffle = True, random_state = random_state)
        model = SVC(random_state = random_state, probability = True)
        param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf', 'linear']}
        gsearch = GridSearchCV(model, param_grid, scoring = 'balanced_accuracy', n_jobs = n_jobs, cv = cv_hyperparam, refit = True)
        result = gsearch.fit(X_train, y_train)
        best_model = result.best_estimator_
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_true = y_test, y_pred = y_pred)
        scores.append(acc)
        
    scores = pd.DataFrame(np.asarray(scores), index = np.shape(scores)[0]*['accuracy'])

    return predicted_features, scores

def svm_svr(adata = None,
            X = None,
            feature_names = None,
            fs_method = None,
            fs_method_params = None,
            feature_threshold = None,
            labels: list = None,
            labels_key: str = None, 
            random_state: int = 0,
            n_splits: int = 10,
            n_jobs: int = -1, 
            **args):
    """Performs SVM regression

    Parameters
    adata: anndata.AnnData (default = None)
        annotated data object containing single-cell data (dimensions = cells x features)
    X: np.ndarray (default = None)
        data (dimensions = cells x features). If None, will access from adata.X
    feature_names: list (default = None)
        list containing feature names. If None, will access from adata.var_names
    fs_method: delve_benchmark.tl.function (default = None)
        function specifying the type of feature selection to perform
    fs_method_params: dict (default = None)
        dictionary specifying the input parameters to the feature selection method
    feature_threshold: int (default = None)
        threshold for selecting for features for evaluation. If None, uses all features
    labels: list (default = None)
        list containing ground truth continuous values for regression
    labels_key: str (default = None)
        string referring to the key in adata object with ground truth continuous values. This will be used if labels is unspecified
    random_state: int (default = 0)
        integer of random state for reproducibility
    n_splits: int (default = 10)
        number of random splits for kfold cross validation
    n_jobs: int (default = -1)
        integer referring to the number of jobs for parallelization
    ----------

    Returns
    predicted_features: np.ndarray
        array of features following feature selection from last fold
    scores: pd.core.frame.DataFrame
        dataframe containing the MSE following regression
    ----------
    """
    if X is None:
        X = adata.X.copy()
        if scipy.sparse.issparse(X):
            X = X.todense()
        feature_names = np.asarray(adata.var_names)
    if labels is None:
        y = np.asarray(adata.obs[labels_key].values).astype(float)
    else:
        y = labels.copy()

    scores = []
    cv = KFold(n_splits = n_splits, shuffle = True, random_state = random_state)
    for train_ix, test_ix in cv.split(X, y):
        X_train, X_test = X[train_ix, :], X[test_ix, :]
        y_train, y_test = y[train_ix], y[test_ix]

        #perform feature selection on training data 
        predicted_features = fs_method(adata = adata[train_ix, :], X = X_train, feature_names = feature_names, n_jobs = n_jobs, **fs_method_params)

        if type(predicted_features) is not pd.core.frame.DataFrame:
            predicted_features = pd.DataFrame(index = predicted_features)  
        
        if feature_threshold is not None:
            predicted_features = np.asarray(predicted_features.index[:feature_threshold])
        else:
            predicted_features = np.asarray(predicted_features.index)

        #subset data according to predicted features 
        X_train = X_train[:, np.isin(feature_names, predicted_features)]
        X_test = X_test[:, np.isin(feature_names, predicted_features)]

        cv_hyperparam = KFold(n_splits = 3, shuffle = True, random_state = random_state)
        model = SVR()
        param_grid = {'C': [0.1, 1, 10, 100], 'epsilon': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf', 'linear']}
        gsearch = GridSearchCV(model, param_grid, scoring = 'neg_mean_squared_error', n_jobs = n_jobs, cv = cv_hyperparam, refit = True)
        result = gsearch.fit(X_train, y_train)
        best_model = result.best_estimator_
        y_pred = best_model.predict(X_test)
        metric_values = mean_squared_error(y_true = y_test, y_pred = y_pred)
        scores.append(metric_values)
        
    scores = pd.DataFrame(np.asarray(scores), index = np.shape(scores)[0]*['mean_squared_error'])

    return predicted_features, scores