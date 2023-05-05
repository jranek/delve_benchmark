import pandas as pd
import numpy as np
import scanpy as sc 
import anndata
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def standardize(x: np.ndarray = None):
    """Standardizes data by removing the mean and scaling to unit variance.

    Parameters
    x: np.ndarray (default = None)
        data matrix (dimensions = cells x features)
    ----------

    Returns
    X: np.ndarray 
        standardized data matrix (dimensions = cells x features)
    ----------
    """
    scaler = StandardScaler(with_mean = True, with_std = True)
    X = scaler.fit_transform(x)
    return X

def min_max_scale(x: np.ndarray = None):
    """Normalize data by min max scaling

    Parameters
    x: np.ndarray (default = None)
        data matrix (dimensions = cells x features)
    ----------

    Returns
    X: np.ndarray 
        normalized data matrix (dimensions = cells x features)
    ----------
    """
    scaler = MinMaxScaler()
    X = scaler.fit_transform(x)
    return X

def preprocess_rpe(df: pd.core.frame.DataFrame = None,
                    batch_df: pd.core.frame.DataFrame = None,
                    batch_id: str = None,
                    remove_unlabel: bool = True,
                    remove_area: bool = True,
                    remove_derived: bool = True,
                    batch_correct: bool = True):
    """Preprocesses RPE data by:
        1. subsetting feature set (removes area features and derived features if specified)
        2. subsetting cells with ground truth labels
        3. min max scale for normalization
        4. batch effect correcting

    Parameters
    df: pd.core.Frame.DataFrame (default = None)
        DataFrame containing single-cell data (dimensions: cells x features)
    batch_df: pd.core.Frame.DataFrame (default = None)
        DataFrame containing batch information if applicable
    batch_id: str (default = None)
        String of column within batch_df
    remove_unlabel: bool (default = True)
        Boolean, if True, only cells with ground truth labels are returned
    remove_area: bool (default = True)
        Boolean, if True, remove areashape features apart from area 
    remove_derived: bool (default = True)
        Boolean, if True, remove derived ratio features
    batch_correct: bool (default = True)
        Boolean, if True, perform batch effect correction
    ----------

    Returns
    df_X: pd.core.Frame.DataFrame:
        DataFrame containing normalized single-cell data
    df_state: pd.core.Frame.DataFrame:
        DataFrame obtaining all observation metadata
    df_cleaned: pd.core.Frame.DataFrame
        DataFrame containing original subsetted data
    y_batch: np.ndarray
        Array containing batch information
    ----------
    """
    df_state = df.loc[:, ['age','annotated_age', 'phase', 'annotated_phase', 'state']]
    df_cleaned = df.drop(columns = ['annotated_age','predicted_age', 'age', 'phase', 'annotated_phase', 'cyto_over_DNA', 'state'])
    df_cleaned = df_cleaned.loc[:, ~df_cleaned.columns.str.startswith('Unnamed:')]
    df_cleaned = df_cleaned.loc[:, ~df_cleaned.columns.str.startswith('Int_Intg')]
    df_cleaned['Int_Intg_DNA_nuc'] = df['Int_Intg_DNA_nuc'].copy()

    if remove_area == True:    
        df_cleaned = df_cleaned.loc[:, ~df_cleaned.columns.str.startswith('AreaShape')]
        df_cleaned['AreaShape_Area_nuc'] = df['AreaShape_Area_nuc'].copy()
    
    if remove_derived == True:
        df_cleaned = df_cleaned.loc[:, ~df_cleaned.columns.str.contains('_over_')]

    if remove_unlabel == True:
        phase_idx = np.where(~df_state['annotated_phase'].isnull().values == True)[0]
        M_idx = np.where(df_state['annotated_phase'].values != 'M')[0] #too few cells so we'll remove this
        phase_idx = np.array(list(set(phase_idx).intersection(set(M_idx))))
        age_idx = np.where(~df_state['annotated_age'].isnull().values == True)[0]
        ground_truth_idx = np.array(list(set(phase_idx).intersection(set(age_idx))))

        df_cleaned = df_cleaned.iloc[ground_truth_idx, :]
        df_state = df_state.iloc[ground_truth_idx, :]
        if batch_df is not None:
            batch_df = batch_df.iloc[ground_truth_idx, :]

    if batch_df is not None:
        y_batch = batch_df[batch_id].values.astype('str')

    df_X = pd.DataFrame(min_max_scale(np.asarray(df_cleaned)), columns = df_cleaned.columns, index = df_cleaned.index) #min max normalized

    if batch_correct == True:
        adata_pp = anndata.AnnData(df_X)
        adata_pp.obs['batch'] = y_batch.copy()
        corrected = sc.pp.combat(adata_pp, inplace=False)
        df_X = pd.DataFrame(corrected, columns = df_X.columns)
    else:
        y_batch = None

    return df_X, df_state, df_cleaned, y_batch

def create_rpe_adata(df = None,
                    batch_df = None,
                    batch_id = 'well', 
                    remove_unlabel: bool = True,
                    remove_area: bool = True, 
                    remove_derived: bool = True,
                    batch_correct: bool = True):
    """Preprocesses RPE data and creates adata object with all appended information:
        1. subsetting feature set (removes area features and derived features if specified)
        2. subsetting cells according to ground truth
        3. min max scale for normalization
        4. batch effect correcting

    Parameters
    df: pd.core.Frame.DataFrame (default = None)
        DataFrame containing single-cell data (dimensions: cells x features)
    batch_df: pd.core.Frame.DataFrame (default = None)
        DataFrame containing batch information if applicable
    batch_id: str (default = None)
        String of column within batch_df
    remove_unlabel: bool (default = True)
        Boolean, if True, only cells with ground truth labels are returned
    remove_area: bool (default = True)
        Boolean, if True, remove areashape features apart from area 
    remove_derived: bool (default = True)
        Boolean, if True, remove derived ratio features
    batch_correct: bool (default = True)
        Boolean, if True, perform batch effect correction
    ----------

    Returns
    adata: anndata.AnnData (default = None)
        Anndata object containing 4i data and all necessary appended observations
    ----------
    """
    df_X, df_state, df_cleaned, y_batch = preprocess_rpe(df = df, batch_df = batch_df, batch_id=batch_id,
                                                        remove_unlabel = remove_unlabel, remove_area = remove_area,
                                                        remove_derived = remove_derived, batch_correct = batch_correct)

    y_annot_phase = df_state['annotated_phase'].values.astype('str')
    y_age = df_state['annotated_age'].values
    y_state = df_state['state'].values.astype('str')
    prolif_ind = np.where(y_state == 'proliferative')[0] #only renaming this to make comparable to PDAC data fyi
    y_state[prolif_ind] = 'cycling'

    y_phase = y_annot_phase.copy()
    idx_G0 = np.where(y_state == 'arrested') or np.where(y_phase == 'G1')
    y_phase[idx_G0[0]] = 'G0'

    adata = anndata.AnnData(df_X)
    adata.obs['annotated_phase'] = y_annot_phase.copy()
    adata.obs['annotated_age'] = y_age.copy()
    adata.obs['state'] = y_state.copy()
    adata.obs['phase'] = y_phase.copy()

    if y_batch is not None:
        adata.obs['batch'] = y_batch.copy()

    adata.layers['raw'] = df_cleaned.copy()

    return adata

def preprocess_pdac(df = None,
                    cellline: str = None,
                    condition: str = None,
                    remove_unlabel: bool = True,
                    remove_area: bool = True,
                    remove_derived: bool = True):
    """Preprocesses PDAC data by:
        1. subsetting feature set (removes area features and derived features if specified)
        2. subsetting cells according to cellline
        2. subsetting cells according to condition label
        3. min max scale for normalization

    Parameters
    df: pd.core.Frame.DataFrame (default = None)
        DataFrame containing single-cell data (dimensions: cells x features)
    cellline: str (default = None)
        String referring to the cellline by which subset cells
    condition: str (default = None)
        String referring to the condition by which subset cells
    remove_unlabel: bool (default = True)
        Boolean, if True, only cells with ground truth labels are returned
    remove_area: bool (default = True)
        Boolean, if True, remove areashape features apart from area 
    remove_derived: bool (default = True)
        Boolean, if True, remove derived ratio features
    ----------

    Returns
    df_X: pd.core.Frame.DataFrame:
        DataFrame containing normalized single-cell data
    df_state: pd.core.Frame.DataFrame:
        DataFrame obtaining all observation metadata
    df_cleaned: pd.core.Frame.DataFrame
        DataFrame containing original subsetted data
    ----------
    """
    df_state = df.loc[:, ['WellID','condition', 'state', 'DNA_content', 'phase', 'genome', 'cellline']]
    df_cleaned = df.drop(columns = ['WellID','condition', 'state', 'DNA_content', 'phase', 'genome', 'cellline', 'PHATE_1', 'PHATE_2', 'PHATE3d_1', 'PHATE3d_2', 'PHATE3d_3'])
    df_cleaned = df_cleaned.loc[:, ~df_cleaned.columns.str.startswith('Unnamed:')]
    df_cleaned = df_cleaned.loc[:, ~df_cleaned.columns.str.startswith('Int_Intg')]
    df_cleaned['Int_Intg_DNA_nuc'] = df['Int_Intg_DNA_nuc'].copy()

    if remove_area == True:    
        df_cleaned = df_cleaned.loc[:, ~df_cleaned.columns.str.startswith('AreaShape')]
        df_cleaned['AreaShape_Area_nuc'] = df['AreaShape_Area_nuc'].copy()
    
    if remove_derived == True:
        df_cleaned = df_cleaned.loc[:, ~df_cleaned.columns.str.contains('_over_')]

    if remove_unlabel == True:
        phase_idx = np.where(~df_state['phase'].isnull().values == True)[0]
        df_cleaned = df_cleaned.iloc[phase_idx, :]
        df_state = df_state.iloc[phase_idx, :]

    cell_idx = np.where(df_state['cellline'] == cellline)[0]
    df_cleaned = df_cleaned.iloc[cell_idx, :]
    df_state = df_state.iloc[cell_idx, :]

    if condition is not None:
        condition_idx = np.where(df_state['condition'] == condition)[0]
        df_cleaned = df_cleaned.iloc[condition_idx, :]
        df_state = df_state.iloc[condition_idx, :]

    df_X = pd.DataFrame(min_max_scale(np.asarray(df_cleaned)), columns = df_cleaned.columns, index = df_cleaned.index) #min max normalized

    return df_X, df_state, df_cleaned

def create_pdac_adata(df = None,
                    cellline = None,
                    condition = None,
                    remove_unlabel: bool = True,
                    remove_area: bool = True, 
                    remove_derived: bool = True):
    """Preprocesses PDAC data and creates adata object with all appended information:
        1. subsetting feature set (removes area features and derived features if specified)
        2. subsetting cells according to cellline and condition
        3. min max scale for normalization

    Parameters    
    df: pd.core.Frame.DataFrame (default = None)
        DataFrame containing single-cell data (dimensions: cells x features)
    cellline: str (default = None)
        String referring to the cellline by which subset cells
    condition: str (default = None)
        String referring to the condition by which subset cells
    remove_unlabel: bool (default = True)
        Boolean, if True, only cells with ground truth labels are returned
    remove_area: bool (default = True)
        Boolean, if True, remove areashape features apart from area 
    remove_derived: bool (default = True)
        Boolean, if True, remove derived ratio features
    ----------

    Returns
    adata: anndata.AnnData (default = None)
        Anndata object containing 4i data and all necessary appended observations
    ----------
    """
    df_X, df_state, df_cleaned = preprocess_pdac(df = df, cellline = cellline,condition = condition,
                                                remove_unlabel = remove_unlabel, remove_area = remove_area,
                                                remove_derived = remove_derived)

    y_phase = df_state['phase'].values.astype('str')
    y_condition = df_state['condition'].values.astype('str')
    y_state = df_state['state'].values.astype('str')
    y_cellline = df_state['cellline'].values.astype('str')
    y_DNA_content = df_state['DNA_content'].values.astype('str')

    adata = anndata.AnnData(df_X)
    adata.obs['phase'] = y_phase.copy()
    adata.obs['condition'] = y_condition.copy()
    adata.obs['state'] = y_state.copy()
    adata.obs['cellline'] = y_cellline.copy()
    adata.obs['DNA_content'] = y_DNA_content.copy()

    adata.layers['raw'] = df_cleaned.copy()

    return adata