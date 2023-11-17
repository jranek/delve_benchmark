import pandas as pd
import numpy as np
import scanpy as sc 
import anndata
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import scvelo as scv
import delve_benchmark

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
    adata: anndata.AnnData
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
    adata: anndata.AnnData 
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

def filter_DE(adata,
            min_genes = 650,
            max_genes = 7200,
            percent_mito = 0.2,
            min_shared_counts = 5,
            min_counts = 1500,
            max_counts = 90000):
    """Performs QC filtering on the definitive endoderm dataset by filtering cells and genes according to appropriate cutoffs

    Parameters    
    adata: anndata.AnnData
        anndata object containing the single-cell RNA sequencing data
    min_genes: int (default = 650)
        minumum number of genes expressed within a cell
    max_genes: int (default = 7200)
        maximum number of genes expressed within a cell
    percent_mito: float (default = 0.2)
        proportion of mitochondrial genes expressed
    min_shared_counts: int (default = 5)
        minumum number of unspliced and spliced molecular counts
    min_counts: int (default = 1500)
        minumum number of gene counts 
    max_counts: int (default = 90000)
        maximum number of gene counts
    ----------

    Returns
    adata: anndata.AnnData
        Filtered anndata object 
    ----------
    """
    sc.pp.filter_cells(adata, min_genes = min_genes)
    sc.pp.filter_cells(adata, max_genes = max_genes)
    adata = adata[adata.obs['percent_mito'] <= percent_mito]
    scv.pp.filter_genes(adata, min_shared_counts = min_shared_counts)
    sc.pp.filter_cells(adata, min_counts = min_counts)
    sc.pp.filter_cells(adata, max_counts = max_counts)

    return adata

def demultiplex(adata, mat, barcodes, gene_names, numBarcodes = 3):
    ####Performs de-multiplexing by linking the maximum median normalized condition-specific barcode to each cell
    fulldf = pd.DataFrame(mat.todense(), columns = barcodes, index = gene_names)
    bcddf = fulldf.iloc[-numBarcodes:len(fulldf), :]
    median_bcd = bcddf.median(axis = 1)
    mednorm_bcd = bcddf.div(median_bcd, axis = 'index') #median normalized data
    bcd_df = mednorm_bcd.copy() #converts downstream analysis into looking at median normalized data 
    barcode_groups = bcd_df.idxmax()

    barcodeReind = sorted(range(len(adata.obs_names)), key=lambda k: adata.obs_names[k]) #gives index of alphabetical cell barcodes
    Z = [x for _,x in sorted(zip(barcodeReind, barcode_groups))] #reoder feature barcodes by median normalized max
    return Z
    
def preprocess_DE(adata,
                mat,
                barcodes,
                gene_names,
                numBarcodes = 3,
                barcode_dict = {'FtrBarcode1': 'd0','FtrBarcode2': 'd1', 'FtrBarcode3': 'd2'},
                min_genes = 650,
                max_genes = 7200,
                percent_mito = 0.2,
                min_shared_counts = 5,
                min_counts = 1500,
                max_counts = 90000,
                n_top_genes = 2000,
                n_neighbors = 10,
                n_pcs = 50,
                groupby = 'day',
                n_jobs = -1):
    """Preprocesses DE data by:
        1. QC filtering
        2. de-mulitplexing
        3. CPM normalization + log+1 transformation
        4. HVG selection
        5. RNA velocity estimation

    Parameters    
    adata: anndata.AnnData
        anndata object containing the single-cell RNA sequencing data
    mat, barcodes, gene_names:
        output from 10X Genomics filtered_feature_bc_matrix 
    numBarcodes: int (default = 3)
        number of condition-specific oligo barcodes for demulitplexing
    barcode_dict: dict 
        dictionary mapping the feature barcode labels to condition-specific labels 
    min_genes: int (default = 650)
        minumum number of genes expressed within a cell
    max_genes: int (default = 7200)
        maximum number of genes expressed within a cell
    percent_mito: float (default = 0.2)
        proportion of mitochondrial genes expressed
    min_shared_counts: int (default = 5)
        minumum number of unspliced and spliced molecular counts
    min_counts: int (default = 1500)
        minumum number of gene counts 
    max_counts: int (default = 90000)
        maximum number of gene counts
    n_top_genes: int (default = 2000)
        number of highly variable genes 
    n_neighbors: int (default = 10)
        number of k-nearest neighbors
    n_pcs: int (default = 50)
        number of principle components
    groupby: str (default = 'day')
        labels key for RNA velocity differential kinetics correction
    n_jobs: int (default = -1)
        number of parallel processes 
    ----------

    Returns
    adata: anndata.AnnData
        preprocessed anndata object 
    ----------
    """
    cell_barcodes = demultiplex(adata, mat, barcodes, gene_names, numBarcodes = numBarcodes)
    adata.obs['barcode_groups'] = pd.Categorical(cell_barcodes)
    adata.obs['day'] = pd.Series(adata.obs['barcode_groups']).map(barcode_dict)

    adata.obs['n_counts'] = adata.X.sum(1)
    adata.obs['n_genes'] = (adata.X > 0).sum(1)
    mito_genes = adata.var_names.str.startswith('MT-')
    adata.obs['percent_mito'] = np.sum(adata[:, mito_genes].X, axis = 1).A1 / np.sum(adata.X, axis = 1).A1

    sc.pl.violin(adata, ['n_genes', 'n_counts', 'percent_mito'], jitter = 0.4, multi_panel = True)
    sc.pl.scatter(adata, x='n_counts', y='percent_mito')
    sc.pl.scatter(adata, x='n_counts', y='n_genes')

    print(adata.shape)

    adata = filter_DE(adata, min_genes = min_genes, max_genes = max_genes, percent_mito = percent_mito, min_shared_counts = min_shared_counts, min_counts = min_counts, max_counts = max_counts)

    sc.pl.violin(adata, ['n_genes', 'n_counts', 'percent_mito'], jitter = 0.4, multi_panel = True)
    sc.pl.scatter(adata, x='n_counts', y='percent_mito')
    sc.pl.scatter(adata, x='n_counts', y='n_genes')

    print(adata.shape)

    adata.layers['raw'] = adata.X.copy()

    sc.pp.normalize_total(adata, target_sum = 1e6)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, flavor = 'seurat', n_top_genes = n_top_genes, subset = True)

    sc.pp.neighbors(adata, n_neighbors = n_neighbors)
    adata = delve_benchmark.tl.perform_velocity_estimation(adata = adata, n_pcs = n_pcs, k = n_neighbors, var_names = 'all', mode = 'dynamical', groupby = groupby, likelihood_threshold = 0.001, n_jobs = n_jobs)
    return adata