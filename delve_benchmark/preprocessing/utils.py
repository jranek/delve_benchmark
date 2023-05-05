import os
import types
import pkg_resources
import scanpy as sc
import anndata
import scipy
import numpy as np

def _get_imports():
    for name, val in globals().items():
        if isinstance(val, types.ModuleType):
            name = val.__name__.split(".")[0]

        elif isinstance(val, type):
            name = val.__module__.split(".")[0]
            
        rename_packages = {"PIL": "Pillow",
                            "sklearn": "scikit-learn"}

        if name in rename_packages.keys():
            name = rename_packages[name]

        yield name

def print_imports():
    """Prints import statements of the loaded in Python modules.

    Parameters
    ----------

    Returns
    -------
    """
    imports = list(set(_get_imports()))
    requirements = []
    for m in pkg_resources.working_set:
        if m.project_name in imports and m.project_name!="pip":
            requirements.append((m.project_name, m.version))

    for r in requirements:
        print("{}=={}".format(*r))

    sc.logging.print_version()

def make_directory(directory: str = None):
    """Creates a directory at the specified path if one doesn't exist.

    Parameters
    ----------
    directory : str
        A string specifying the directory path.

    Returns
    -------
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def parse_input(adata: anndata.AnnData):
    """Accesses and parses data from adata object
    Parameters
    adata: anndata.AnnData
        annotated data object where adata.X is the attribute for preprocessed data
    ----------
    Returns
    X: np.ndarray
        array of data (dimensions = cells x features)
    feature_names: np.ndarray
        array of feature names
    obs_names: np.ndarray
        array of cell names   
    ----------
    """
    try:
        if isinstance(adata, anndata.AnnData):
            X = adata.X.copy()
        if isinstance(X, scipy.sparse.csr_matrix):
            X = np.asarray(X.todense())

        feature_names = np.asarray(adata.var_names)
        obs_names = np.asarray(adata.obs_names)
        return X, feature_names, obs_names
    except NameError:
        return None