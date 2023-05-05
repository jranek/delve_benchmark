import sys
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
import delve_benchmark

class fs(BaseEstimator):
    """Class for evaluating feature selection methods on trajectory preservation

        Parameters
        ----------------------------
        adata: anndata.AnnData (default = None)
            annotated data object containing preprocessed single-cell data (dimensions = cells x features)
        X: np.ndarray (default = None)
            array containing preprocessed single-cell data. If None, will access from adata.X
        feature_names: list (default = None)
            list of feature names. If None, will access from adata.var_names
        predicted_features: np.ndarray (default = None)
            array containing selected features following feature selection. If None, will perform feature selection according to specified feature selection method function
        reference_features: np.ndarray (default = None)
            array containing reference features for evaluation. If None, will perform random forest classification using provided labels 
        feature_threshold: int (default = 30)
            number of features to select from a feature selection strategy
        feature_threshold_reference: int (default = 30)
            number of features to select from a reference list of features 
        labels_key: str (default = None)
            string referring to the key in adata.obs of ground truth labels
        labels: (default = None)
            array referring to the labels for every cell
        ref_labels: (default = None)
            array containing cell type labels if using random forest classification for evaluation
        fs_method: function (default = None)
            function housed in the delve_benchmark.tl.feature_selection script that specifies the feature selection strategy to perform. Can be one of the following (or you may provide your own):
                delve_benchmark.tl.delve_fs
                delve_benchmark.tl.random_forest
                delve_benchmark.tl.laplacian_score_fs
                delve_benchmark.tl.neighborhood_variance_fs
                delve_benchmark.tl.mcfs_fs
                delve_benchmark.tl.scmer_fs
                delve_benchmark.tl.hotspot_fs
                delve_benchmark.tl.hvg
                delve_benchmark.tl.variance_score
                delve_benchmark.tl.seed_features
                delve_benchmark.tl.all_features
                delve_benchmark.tl.random_features
        fs_method_params: dictionary (default = None)
            dictionary referring to the feature selection method hyperparameters. For more information on method-specific hyperparameters, see the delve_benchmark.tl.feature_selection script for the method of interest. Can be:
                delve_benchmark.tl.delve_fs example: {'num_subsamples': 1000, 'n_clusters': 5, 'k': 10, 'n_random_state': 10}
                delve_benchmark.tl.random_forest example: {'n_splits': 10, 'labels_key': 'phase'}
                delve_benchmark.tl.laplacian_score_fs example: {'k': 10}
                delve_benchmark.tl.neighborhood_variance_fs example: None
                delve_benchmark.tl.mcfs_fs example: {'k': 10, 'n_selected_features': 30, 'n_clusters': 4}
                delve_benchmark.tl.scmer_fs example: {'k': 10, 'n_pcs': 50}
                delve_benchmark.tl.hotspot_fs example: {'k': 10, 'n_pcs': 50}
                delve_benchmark.tl.hvg example: {'n_top_genes': 30, 'log': True}
                delve_benchmark.tl.variance_score example: None
                delve_benchmark.tl.seed_features example: {'num_subsamples': 1000, 'n_clusters': 5, 'k': 10, 'n_random_state': 10}
                delve_benchmark.tl.all_features example: None
                delve_benchmark.tl.random_features example: None
        eval_method: function (default = None)
            function housed in the delve_benchmark.tl.evaluate script that specifies the evaluation method to perform. Can be one of the following (or you may provide your own):
                precision @ k: delve_benchmark.tl.pak
                support vector machine classification: delve_benchmark.tl.svm
                support vector machine regression: delve_benchmark.tl.svm_svr
                kendall rank correlation following trajectory inference: delve_benchmark.tl.kt_corr
                kmeans clustering: delve_benchmark.tl.cluster
        eval_method_params: dictionary (default = None)
            dictionary referring to the evaluation method hyperparameters. For more information on evaluation method-specific hyperparameters, see the delve_benchmark.tl.evaluate script for the method of interest. Can be:
                delve_benchmark.tl.pak example: {'k_sweep': [5, 10, 15, 20, 25, 30]}
                delve_benchmark.tl.svm example: {'n_splits': 10}
                delve_benchmark.tl.svm_svr example: {'n_splits': 10}
                delve_benchmark.tl.kt_corr example: {'roots_arr': [[354, 13, 22, 382, 9, 54, 99, 239, 10, 87], [983, 41, 96, 1464, 541, 1520, 818, 739, 1359, 1868]], 'corr_reference_features': False}
                delve_benchmark.tl.cluster example: {'n_clusters': 4, 'n_sweep': 25}                
        n_jobs: int (default = 1)
            number of jobs to use in computation 

        Attributes
        ----------------------------
        model.select()
            performs feature selection according to a specified feature selection method

            Returns:
                predicted features: ranked list of features following feature selection
                delta_mean: dataframe containing pairwise change in expression across prototypical cellular neighborhoods (only for DELVE feature selection if 'return_modules' = True in fs_method_params) 
                modules: dataframe containing feature-cluster assignment (only for DELVE feature selection if 'return_modules' = True in fs_method_params)

        model.evaluate_select()
            performs feature selection according to a feature selection strategy and then evaluates method according to the evaluation criteria or task of interest

            Returns:
                predicted features: ranked list of features following feature selection
                score_df: dataframe of evaluated scores

        Examples
        ----------------------------
        1. Example for DELVE feature selection:

            fs = delve_benchmark.tl.fs(adata = adata, X = adata.X, feature_names = adata.var_names, fs_method = delve_benchmark.tl.delve_fs, fs_method_params = {'num_subsamples': 1000, 'n_clusters': 5, 'k': 10, 'n_random_state': 10, 'return_modules': True})
            delta_mean, modules, predicted_features = fs.select()
        
        2. Example for laplacian score feature selection:

            fs = delve_benchmark.tl.fs(adata = adata, X = adata.X, feature_names = adata.var_names, fs_method = delve_benchmark.tl.laplacian_score_fs, fs_method_params = {'k': 10})
            predicted_features = fs.select()

        2. Example for SVM classification following feature selection with DELVE:

            fs = delve_benchmark.tl.fs(adata = adata, X = adata.X, feature_names = adata.var_names, fs_method = delve_benchmark.tl.delve_fs,
                            fs_method_params = {'num_subsamples': 1000, 'n_clusters': 5, 'k': 10, 'n_random_state': 10},
                            eval_method = delve_benchmark.tl.svm, eval_method_params = {'n_splits': 10}, labels_key = 'phase', feature_threshold = 30)
            predicted_features, scores = fs.evaluate_select()
            
        3. Example clustering evaluation using hotspot:
            fs = delve_benchmark.tl.fs(adata = adata, X = adata.X, feature_names = adata.var_names, fs_method = delve_benchmark.tl.hotspot_fs, fs_method_params = {'k': 10, 'n_pcs': 50},
                                        eval_method = delve_benchmark.tl.cluster, eval_method_params =  {'n_clusters': 4, 'n_sweep': 25}, labels_key = 'phase', feature_threshold = 30)
            predicted_features, scores = fs.evaluate_select()
        ----------
    """
    def __init__(
        self,
        adata = None,
        X = None,
        feature_names = None,
        predicted_features = None,
        reference_features = None,
        feature_threshold = 30,
        feature_threshold_reference = 30,
        fs_method = None,
        fs_method_params = None,
        eval_method = None,
        eval_method_params = None,
        labels_key = None,
        labels = None,
        ref_labels = None,
        n_jobs = -1,
        **fs_kwargs
    ):
        
        self.adata = adata
        self.X = X
        self.feature_names = feature_names
        self.predicted_features = predicted_features
        self.reference_features = reference_features
        self.feature_threshold = feature_threshold
        self.feature_threshold_reference = feature_threshold_reference
        self.fs_method = fs_method
        self.fs_method_params = fs_method_params
        self.eval_method = eval_method
        self.eval_method_params = eval_method_params
        self.fs_kwargs = fs_method_params
        self.labels_key = labels_key
        self.labels = labels
        self.ref_labels = ref_labels
        self.n_jobs = n_jobs
        
        if self.fs_kwargs is None:
            self.fs_kwargs = {}

        if self.fs_method_params is None:
            self.fs_method_params = {}

        if self.eval_method_params is None:
            self.eval_method_params = {}
        
    def evaluate_select(self):
        if (self.eval_method.__name__ == 'svm') or (self.eval_method.__name__ == 'svm_svr'): 
            # if evaluation is classification or regression, then call evaluation function directly. In this case, feature selection will be performed inside the test/train split
            sys.stdout.write('performing evaluation: {}'.format(self.eval_method.__name__)+'\n')
            self.predicted_features, self.scores_df = self._evaluate()
        
        else:
            # for all other evaluation criteria: First: perform feature selection, Second: perform evaluation
            sys.stdout.write('feature selection method: {}'.format(self.fs_method.__name__)+'\n')
            sys.stdout.write('performing feature selection with parameters: {}'.format(self.fs_method_params)+'\n')

            if self.predicted_features is None:
                # select features according to a feature selection strategy
                predicted_features = self.select()

                # convert output of feature selection method to the same data structure
                if type(predicted_features) is not pd.core.frame.DataFrame:
                    predicted_features = pd.DataFrame(index = predicted_features)
                if self.feature_threshold is not None:
                    self.predicted_features = np.asarray(predicted_features[:self.feature_threshold].index) #select top x features
                else:
                    self.predicted_features = np.asarray(predicted_features.index)

            if (self.reference_features is None and self.eval_method.__name__ != 'cluster'):
                # if evaluation requires comparison to the random forest classifier, then obtain random forest reference features
                sys.stdout.write('obtaining reference features using random forest'+'\n')

                reference_features = delve_benchmark.tl.random_forest(adata = self.adata, X = self.X, feature_names = self.feature_names, return_mean = True,
                                                                labels = self.ref_labels, n_jobs = self.n_jobs, n_splits = 10)

                if self.feature_threshold_reference is not None:
                    self.reference_features = np.asarray(reference_features.sort_values(by = 'random_forest', ascending = False)[:self.feature_threshold_reference].index)
                else:
                    self.reference_features = np.asarray(reference_features.sort_values(by = 'random_forest', ascending = False).index)

            #perform evaluation according to an evaluation metric
            sys.stdout.write('performing evaluation: {}'.format(self.eval_method.__name__)+'\n')
            self.scores_df = self._evaluate()

        self._aggregate() 
        return self.predicted_features, self.scores_df

    def select(self): #performs feature selection according to specified feature selection method and feature selection parameters
        adata_run = self.adata.copy()
        predicted_features = self.fs_method(adata = adata_run, X = adata_run.X, feature_names = self.feature_names, n_jobs = self.n_jobs, **self.fs_kwargs)
        return predicted_features
    
    def _evaluate(self): #performs evaluation according to specified evaluation function and evaluation parameters
        score_df = self.eval_method(adata = self.adata, X = self.X, feature_names = self.feature_names, fs_method = self.fs_method,
                                    fs_method_params = self.fs_method_params, feature_threshold = self.feature_threshold,
                                    predicted_features = self.predicted_features, reference_features = self.reference_features,
                                    labels_key = self.labels_key, labels = self.labels, n_jobs = self.n_jobs, **self.eval_method_params)
        return score_df
    
    def _aggregate(self):
        p = str(self.fs_method.__name__)
        self.scores_df.columns = np.shape(self.scores_df)[1]*[p]