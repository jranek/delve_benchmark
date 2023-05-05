import seaborn as sns
import matplotlib.pyplot as plt
import phate
import numpy as np
import os
import numpy as np
import pandas as pd
import phate
import matplotlib.pyplot as plt
import scanpy as sc
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import anndata
import glob
import delve_benchmark

def plot_seed(adata_directory = None,
            feature_directory = None, 
            adata_name = None,
            labels_key = 'phase',
            save_dir = 'figures',
            filename_save = None,
            trial = 0, 
            m_order = ['G0', 'G1', 'S', 'G2', 'M'],
            ylim = [-0.25,0.25],
            reorder = None):
    #plots feature modules (e.g. figure 4a, figure 5a-c) 
    sc.set_figure_params(fontsize=18, figsize=None, color_map=None, format='pdf', facecolor=None, transparent=False, ipython_format='png2x')
    sns.set_style('ticks')
    colors_dict = {'static': '#C9C9C9',
                'dynamic': '#B46CDA',
                'dynamic 0': '#B46CDA',
                'dynamic 1': '#78CE8B',
                'dynamic 2': '#FF8595',
                'dynamic 3': '#1885F2'}

    adata = sc.read(os.path.join(adata_directory, adata_name+'.h5ad'))
    adata.obs[labels_key] = adata.obs[labels_key].cat.reorder_categories(m_order)
    delta_mean = pd.read_csv(os.path.join(feature_directory, f'delta_mean_trial{trial}.csv'), index_col = 0)
    delta_mean_annot = delta_mean.copy()
    delta_mean_annot.index = delta_mean_annot.index.astype('str')
    delta_mean_annot[labels_key] = adata[np.isin(adata.obs_names, delta_mean_annot.index)].obs[labels_key]

    modules = pd.read_csv(os.path.join(feature_directory, f'modules_trial{trial}.csv'), index_col = 0)
    if reorder is not None:
        modules = pd.DataFrame(pd.Series(modules['cluster_id']).map(reorder))

    dyn_feats = np.concatenate([modules.index[modules['cluster_id'].values == i] for i in [i for i in np.unique(modules['cluster_id']) if 'dynamic' in i]])

    #plot heatmap of expression ordered by labels
    ax = sc.pl.matrixplot(adata, dyn_feats, groupby = labels_key, swap_axes=False, cmap='Blues', standard_scale = 'var', colorbar_title='scaled\nexpression', save = filename_save+'.pdf')
    os.rename(os.path.join('figures', 'matrixplot_'+filename_save + '.pdf'), os.path.join(save_dir, 'matrixplot_'+filename_save + '.pdf'))

    #plot dynamic time traces ordered by labels
    clusters = np.unique(modules['cluster_id'].values)
    colors_dict = dict((k, colors_dict[k]) for k in clusters if k in colors_dict)
    sub_idx = len(clusters)
    _, axes = plt.subplots(1, sub_idx, figsize = (4.5*sub_idx, 3.5), gridspec_kw={'hspace': 0.45, 'wspace': 0.4, 'bottom':0.15})
    for i, ax in zip(range(0, sub_idx), axes.flat):
        cluster = np.asarray(modules.index[modules['cluster_id'] == clusters[i]])
        delta_mean_annot_melt = delta_mean_annot.melt(id_vars=[labels_key])
        delta_mean_annot_melt = delta_mean_annot_melt[np.isin(delta_mean_annot_melt.variable.values, cluster)]
        delta_mean_annot_melt[labels_key] = pd.Categorical(delta_mean_annot_melt[labels_key], ordered=True, categories = m_order)
        delta_mean_annot_melt.sort_values(labels_key, inplace=True)

        g = sns.lineplot(labels_key, 'value', data = delta_mean_annot_melt, ax = ax, color = colors_dict[clusters[i]], estimator = 'mean', ci = 'sd',marker='o')
        g.set_xticklabels(m_order, rotation=45, horizontalalignment='right')
        g.tick_params(labelsize=16)
        g.set_ylabel('mean pairwise $\Delta$ expression', fontsize = 16)
        g.set_xlabel(labels_key, fontsize = 16)
        g.set_title(f'{clusters[i]}', fontsize = 16)
        g.set_ylim(ylim[0], ylim[1])

    plt.savefig(os.path.join(save_dir, filename_save + '_seed_dynamics.pdf'), bbox_inches = "tight")

    #plot UMAP of features colored according to DELVE module assignment
    dyn_adata = anndata.AnnData(delta_mean.transpose())
    dyn_adata.obs['clusters'] = pd.Categorical(modules['cluster_id'].astype('str'))
    sc.pp.neighbors(dyn_adata, use_rep = 'X')
    sc.tl.umap(dyn_adata)

    _, ax = plt.subplots(figsize=(5, 4.5))
    g = sns.scatterplot(dyn_adata.obsm['X_umap'][:, 0], dyn_adata.obsm['X_umap'][:, 1], hue = dyn_adata.obs['clusters'].values, palette = colors_dict, ax = ax, s = 50)
    g.tick_params(labelsize=14)
    g.set_xticks([])
    g.set_yticks([])
    g.set_xlabel('UMAP 1', fontsize = 14)
    g.set_ylabel('UMAP 2', fontsize = 14)
    ax.legend(loc = 'upper right')
    plt.savefig(os.path.join(save_dir, filename_save+'_clusters_umap.pdf'), bbox_inches = "tight")

def plot_string_G(modules = None,
                    colors_dict = {'static': '#C9C9C9', 'dynamic': '#B46CDA', 'dynamic 0': '#B46CDA', 'dynamic 1': '#78CE8B', 'dynamic 2': '#FF8595', 'dynamic 3': '#1885F2'}, 
                    module_id = None,
                    filename_save = None):
    """Plots STRING networks using features within a DELVE module (see figure 5b)
    Parameters
    modules: pd.core.Frame.DataFrame (default = None)
        dataframe containing feature-cluster assignment from DELVE seed selection (step 1)
    colors_dict: dict 
        dictionary containing feature label -> hex code color assignment
    module_id: str
        string containing the id of the DELVE module
    filename_save: str (default = None)
        if specified, will save figure
    ----------
    Returns
    ----------
    """    
    import networkx as nx
    clusters = np.unique(modules['cluster_id'].values)
    colors_dict = dict((k, colors_dict[k]) for k in clusters if k in colors_dict)

    selected_feats = list(modules.index[modules['cluster_id'] == module_id])
    interactions = delve_benchmark.tl.compute_string_interaction(selected_feats)
    G = delve_benchmark.tl.compute_G(interactions)

    node_colors = pd.Series(modules['cluster_id']).map(colors_dict)

    pos = nx.spring_layout(G) # position the nodes using the spring layout
    plt.figure(figsize=(8,6)) 
    nx.draw_networkx(G, node_size = 100, node_color = [node_colors.loc[node] for node in G.nodes()], edge_color = '#B8B8B8', font_weight='semibold',font_size='14', width = 1.5,with_labels=True)
    plt.axis('off')
    if filename_save is not None:
        plt.savefig(filename_save+'.pdf', bbox_inches = 'tight')

def plot_phate(adata_directory = None,
                feature_directory = None,
                adata_name = 'adata_RPE',
                methods_arr = None,
                titles = None,
                knn = 30,
                t = 10, 
                colors_dict = None,
                hue_order = None,
                subplots = [3,4],
                figsize = (20, 15),
                n_selected = 30,
                labels_key = 'phase',
                save_dir = 'figures',
                cmap = None, 
                ind = None,
                filename_save = 'PHATE'):
    #plots PHATE for each feature selection strategy specified in methods_arr (e.g. figure 4b, supplementary figures)
    op = phate.PHATE(knn = knn, t = t, n_jobs = -1)
    adata = sc.read(os.path.join(adata_directory, adata_name + '.h5ad'))
    if ind is not None:
        ind = np.where(adata.obs[labels_key] != 'nan')[0]
        adata = adata[ind, :]
    df_X = pd.DataFrame(adata.X, columns = adata.var_names)
    y = np.asarray(adata.obs[labels_key].values)
    method_df = pd.DataFrame()
    for method in methods_arr:
        scores = pd.read_csv(os.path.join(feature_directory, method + '.csv'), index_col = 0)
        if ('all_features' in method) or ('seed_features' in method):
            method_df = pd.concat([pd.DataFrame(scores.values, columns =[method]), method_df], axis = 1)
        elif ('random_features' in method):
            method_df = pd.concat([pd.DataFrame(scores.values[:n_selected], columns =[method]), method_df], axis = 1)
        else:
            method_df = pd.concat([pd.DataFrame(np.asarray(scores.index[:n_selected].values), columns =[method]), method_df], axis = 1)

    _, axes = plt.subplots(subplots[0], subplots[1], figsize = figsize, gridspec_kw={'hspace': 0.4, 'wspace': 0.3, 'bottom':0.15})

    if np.max(subplots) == 1:
        X_subset = df_X.iloc[:, np.isin(df_X.columns, method_df.loc[:, methods_arr[0]])]
        X_phate = op.fit_transform(np.asarray(X_subset))
        if cmap is not None:
            g = sns.scatterplot(X_phate[:, 0], X_phate[:, 1], c = y, ax = axes, s = 10, linewidth = 0, cmap = cmap)
        else:
            g = sns.scatterplot(X_phate[:, 0], X_phate[:, 1], hue = pd.Series(y), ax = axes, s = 10, linewidth = 0, palette = colors_dict, hue_order = hue_order)
        if cmap is not None:
            norm = plt.Normalize(adata.obs[labels_key].min(), adata.obs[labels_key].max())
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            axes.figure.colorbar(sm)
            axes.legend([],[], frameon=False)
        else:
            h,l = axes.get_legend_handles_labels()
            axes.legend(handles=h, labels=hue_order)
            axes.legend(loc = 'upper right')
        g.tick_params(labelsize=14)
        g.set_xticks([])
        g.set_yticks([])
        g.set_xlabel('PHATE 1', fontsize = 14)
        g.set_ylabel('PHATE 2', fontsize = 14)
    else:
        for i, ax in zip(range(0, len(titles)), axes.flat):
            X_subset = df_X.iloc[:, np.isin(df_X.columns, method_df.loc[:, methods_arr[i]])]
            X_phate = op.fit_transform(np.asarray(X_subset))
            g = sns.scatterplot(X_phate[:, 0], X_phate[:, 1],  hue = pd.Series(y), ax = ax, s = 10, linewidth=0, palette = colors_dict, hue_order = hue_order, cmap = cmap)
            g.tick_params(labelsize=14)
            g.set_xticks([])
            g.set_yticks([])
            g.set_title(titles[i], fontsize = 16)
            g.set_xlabel('PHATE 1', fontsize = 14)
            g.set_ylabel('PHATE 2', fontsize = 14)
            h,l = ax.get_legend_handles_labels()
            ax.legend(handles=h, labels=hue_order)
            ax.legend(loc = 'upper right')

    plt.savefig(os.path.join(save_dir, filename_save+'.pdf'), bbox_inches="tight")

def plot_tgraph(result_directory = None,
                filename = None,
                colors_dict = None,
                m_order = None,
                ylim = [0.3, 0.7],
                filename_save = 'trajectory_graph',
                xticklabels = None,
                save_dir = 'figures'):
    #plots trajectory graph jaccard distances in barplot (e.g. supplementary figure 16c bottom left)
    sns.set_style('ticks')
    scores = []
    metrics = pd.read_csv(os.path.join(result_directory, 'trajectory_graph', filename + ".csv"), index_col = 0)
    scores.append(metrics.loc['jacc_dist', :])

    df = pd.DataFrame(pd.DataFrame(metrics.loc['jacc_dist', :]).melt(ignore_index = False))
    df['method'] = [i.split('_trial')[0] for i in df.index]
    fig, axes = plt.subplots(1, 1, figsize = (5.5, 6), gridspec_kw = {'hspace': 0.8, 'wspace': 0.35, 'bottom':0.15})
    g = sns.barplot(data = df, y = 'value', x = 'method', palette = colors_dict, lw = 0., ax = axes, estimator = np.mean, ci = 'sd', capsize = 0.3, order = m_order, errcolor = 'black', errwidth=1)

    for i in axes.containers:
        axes.bar_label(i,fmt='%.2f', fontsize = 12)

    g.set_xticklabels(xticklabels, rotation = 45, horizontalalignment='right')
    g.tick_params(labelsize = 14)
    g.set_xlabel('method', fontsize = 14)
    g.set_ylabel('score', fontsize = 14)
    g.set_title('Jaccard Distance', fontsize = 16)
    g.set_ylim(ylim[0], ylim[1])
    plt.savefig(os.path.join(save_dir, filename_save+'.pdf'), bbox_inches="tight")

    return scores

def plot_heatmap(scores = None, figsize = (5,2.5), vmin = 0, vmax = 1, cmap = 'Purples',m_order = None,
                xlabel = None, ylabel = None, xticklabels = None, save_dir = None, filename_save = None):
    #plots heatmap of ranked scores for PDAC celllines (e.g. supplementary figure 15)           
    sns.set_theme(style="ticks")
    sns.set(font_scale = 1.25)
    fig, ax = plt.subplots(figsize = figsize)
    scores = scores.reindex(m_order)
    g = sns.heatmap(data = scores, cmap = cmap, annot = True, vmin=vmin, vmax=vmax, fmt=".2f")
    g.set_xticklabels(g.get_xticklabels(), rotation=45, horizontalalignment='right')
    g.set_yticklabels(xticklabels, rotation=0)
    g.tick_params(labelsize=15)
    plt.xlabel(xlabel, fontsize = 16)
    plt.ylabel(ylabel, fontsize = 16)
    plt.savefig(os.path.join(save_dir, filename_save+ '.pdf'), bbox_inches = 'tight')

def compute_scores(results_directory = None, metrics = None):
    #compiles scores across csvs from benchmarking runs
    all_results = []
    ktp = 0
    ktl = 0
    for metric in metrics:
        read_directory = os.path.join(results_directory, metric)
        results_df = pd.DataFrame()
        if metric == 'kt_corr_pseudo':
            if ktp == 0:
                ktp += 1
                for i in glob.glob(read_directory + os.sep + '*.csv'):
                    results = pd.read_csv(i, index_col = 0)
                    results = pd.DataFrame(results.iloc[:, 0])
                    results.columns = [i.split(os.sep)[-1].split('_' + metric)[0]]
                    results_df = pd.concat([results_df, results], axis = 1)
            else:
                for i in glob.glob(read_directory + os.sep + '*.csv'):
                    results = pd.read_csv(i, index_col = 0)
                    results = pd.DataFrame(results.iloc[:, 1])
                    results.columns = [i.split(os.sep)[-1].split('_' + metric)[0]]
                    results_df = pd.concat([results_df, results], axis = 1)

        elif metric == 'kt_corr_labels':
            if ktl == 0:
                ktl += 1
                for i in glob.glob(read_directory + os.sep + '*.csv'):
                    results = pd.read_csv(i, index_col = 0)
                    results = pd.DataFrame(results.iloc[:, 0])
                    results.columns = [i.split(os.sep)[-1].split('_' + metric)[0]]
                    results_df = pd.concat([results_df, results], axis = 1)
            else:
                for i in glob.glob(read_directory + os.sep + '*.csv'):
                    results = pd.read_csv(i, index_col = 0)
                    results = pd.DataFrame(results.iloc[:, 1])
                    results.columns = [i.split(os.sep)[-1].split('_' + metric)[0]]
                    results_df = pd.concat([results_df, results], axis = 1)
                    
        else:
            for i in glob.glob(read_directory + '/*.csv'):
                if metric == 'svm':
                    results = pd.read_csv(i, index_col = 0)
                    results = results.loc['accuracy', :]
                else:
                    results = pd.read_csv(i, index_col = 0)
                results.columns = [i.split(os.sep)[-1].split('_' + metric)[0]]
                results_df = pd.concat([results_df, results], axis = 1)
        all_results.append(results_df)
    return all_results

def plot_metrics(all_results = None,
                metrics = None,
                titles = None,
                colors_dict = None,
                subplots = [3,4],
                figsize = (20, 15),
                m_order = None,
                pak_labels = None,
                xticklabels = None,
                save_dir = 'figures',
                filename_save = 'metrics',
                remove = False):
    #plots evaluation metrics (e.g. p@k, NMI, classification accuracy, MSE, pseudotime correlation - figure 3d)            
    sns.set_style('ticks')
    _, axes = plt.subplots(subplots[0],subplots[1], figsize = figsize, gridspec_kw={'hspace': 0.65, 'wspace': 0.3, 'bottom':0.15})
    for i, ax in zip(range(len(all_results)), axes.flat):
        if metrics[i] == 'pak':
            tmp = all_results[i].copy()
            tmp = tmp.melt(ignore_index = False) 
            tmp['variable'] = [i.split('_trial')[0] for i in tmp['variable']]
            tmp['variable'] = pd.Categorical(tmp['variable'],
                                            categories=m_order,
                                            ordered=True)
                        
            g = sns.lineplot(x = tmp.index, y = 'value', hue = 'variable', data = tmp, legend = True, markers=True, marker="o", estimator = np.mean, ci = 'sd', palette=colors_dict, ax = ax)
            g.legend(bbox_to_anchor=(1.13,-0.18), labels = pak_labels, ncol=4, prop={'size':8.5})
            g.tick_params(labelsize=18)
            g.set_xlabel('k', fontsize = 18)
            g.set_ylabel('score', fontsize = 18)
        else:
            all_melt = all_results[i].melt()
            all_melt['variable'] = [i.split('_trial')[0] for i in all_melt['variable']]  
            g = sns.boxplot(x='variable', y='value', data=all_melt, linewidth = 1,fliersize=0, width = 0.6, ax = ax, palette = colors_dict, order = m_order)

            g = sns.stripplot(x='variable', y='value', data=all_melt, linewidth = 0.8, 
                                size=5, edgecolor="black", jitter = True, dodge = True, ax = ax, palette=colors_dict, order = m_order)

            g.set_xticklabels(xticklabels, rotation=45, horizontalalignment='right')
            g.tick_params(labelsize=18)
            g.set_xlabel('', fontsize = 18)
            g.set_ylabel('score', fontsize = 18)
        g.set_title(titles[i], fontsize = 18)
    if remove != False:
        axes[remove[0],remove[1]].set_axis_off() 
    plt.savefig(os.path.join(save_dir, filename_save+'.pdf'), bbox_inches="tight")

    
def plot_simulated(directory, noise, noise_sweep, metric, trajectory_list, cnames_dict, colors_dict, m_order, savedir, filename_save, n_cells, n_genes):
    #plots simulated evaluation metrics
    sns.set_style('ticks')
    for traj in trajectory_list:
        _, axes = plt.subplots(1,3, figsize = (18, 5), gridspec_kw={'hspace': 0.2, 'wspace': 0.3, 'bottom':0.15})
        for i, ax in zip(range(len(noise_sweep)), axes.flat):

            metric_df = pd.read_csv(os.path.join(directory, traj, noise, str(noise_sweep[i]),n_cells,n_genes, metric+'.csv'), index_col = 0)

            metric_df_total = pd.DataFrame()
            for run in np.unique(metric_df.index):
                metric_df_ = metric_df[metric_df.index == run]
                index_name = metric_df_.columns[metric_df_.columns.str.startswith('Unnamed')][0]
                metric_df_.set_index(index_name, drop = True, inplace = True)
                metric_df_.replace('NaN', np.nan, inplace=True)
                metric_df_ = metric_df_.groupby(index_name).mean(0)

                melt_ = metric_df_.melt(ignore_index=False)
                melt_['variable_names'] = [s.split('_trial')[0] for s in melt_['variable']]
                melt_['variable_names'] = melt_['variable_names'].map(cnames_dict)
                metric_df_total = pd.concat([melt_, metric_df_total], axis = 0)

            if metric == 'pak':
                m_order_pak = m_order.copy()
                g = sns.lineplot(x = index_name, y = 'value', data = metric_df_total, hue = 'variable_names', palette = colors_dict, hue_order = m_order_pak, legend = True, markers=True, marker="o", ax = ax, ci= 'sd')
                g.tick_params(labelsize=18)
                g.set_xlabel('k', fontsize = 20)
                g.set_ylabel('precision @ k', fontsize = 20)
                g.set_title('{}: {}'.format(noise, str(noise_sweep[i])), fontsize = 20)
                ax.get_legend().remove()
                ax.set_ylim(0.2, 1)
                ax.set_xlim(20, 200)
            else:
                g = sns.barplot(data = metric_df_total, y = 'value', x = 'variable_names', palette = colors_dict, lw = 0., ax = ax, estimator = np.mean, capsize = 0.3, ci = 'sd', order = m_order, errcolor = 'black', errwidth=1)
                ax.bar_label(ax.containers[0], fmt='%.2f', fontsize = 13, label_type='edge')#, fontweight = 'semibold')

                g.tick_params(labelsize=18)
                g.set_xlabel('method', fontsize = 20)
                if metric == 'acc':
                    g.set_ylabel('kNN classification accuracy', fontsize = 20)
                elif metric == 'kt':
                    g.set_ylabel('Kendall-Tau correlation', fontsize = 20)
                g.set_title('{}: {}'.format(noise, str(noise_sweep[i])), fontsize = 20)
                ax.set_xticks([])
        if metric == 'pak':
           g.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, prop={'size':10})
        plt.savefig(os.path.join(savedir, filename_save+f'_{traj}_metrics.pdf'), bbox_inches="tight")

def compute_medians(directory = None, cnames_dict = None, noise = None, noise_sweep = None, traj = None, metric = None, npak = None, n_cells = None, n_genes = None):
    #computes median score across simulated trajectories
    df = pd.DataFrame()
    for i in range(0, len(noise_sweep)):
        metric_df = pd.read_csv(os.path.join(directory, traj, noise, str(noise_sweep[i]), n_cells, n_genes, metric+'.csv'), index_col = 0)
        metric_df_total = pd.DataFrame()
        for run in np.unique(metric_df.index):
            metric_df_ = metric_df[metric_df.index == run]
            index_name = metric_df_.columns[metric_df_.columns.str.startswith('Unnamed')][0]
            metric_df_.set_index(index_name, drop = True, inplace = True)
            metric_df_.replace('NaN', np.nan, inplace=True)
            metric_df_ = metric_df_.groupby(index_name).mean(0)

            melt_ = metric_df_.melt(ignore_index=False)
            melt_['variable_names'] = [s.split('_trial')[0] for s in melt_['variable']]
            melt_['variable_names'] = melt_['variable_names'].map(cnames_dict)
            metric_df_total = pd.concat([melt_, metric_df_total], axis = 0)

        if metric == 'pak':
            metric_df_total = metric_df_total.loc[npak, :]

        df_ = pd.DataFrame(metric_df_total.groupby('variable_names').median().values, columns = [noise_sweep[i]], index = metric_df_total.groupby('variable_names').median().index.values)
        df = pd.concat([df_, df], axis = 1)

    return df, metric_df_total

def compute_ranking(df, m_order, metric):
    #computes aggregate scores by min max normalizing median scores across simulated trajectories across noise conditions
    if metric == 'pak':
        df.loc['seed'] = np.shape(df)[1]*[np.nan]

    m = df[np.isin(df.index, m_order)]

    scaler = MinMaxScaler()
    m_ranked = pd.DataFrame(scaler.fit_transform(m), index = m.index, columns = m.columns)
    m_ranked = m_ranked.reindex(m_order)

    mean_metrics = pd.DataFrame(m_ranked.mean(1), columns = [metric])

    heatmap_order = list(m_ranked.columns)
    heatmap_order.reverse()
    m_ranked = m_ranked.reindex(columns=heatmap_order)

    return m_ranked, mean_metrics

def plot_ranking(mean_metrics_df, save_dir, filename_save, palette):
    #plots ranked heatmaps from simulation study (e.g. figure 3)
    sns.set_style('white')
    figsize = (4/3, 5)
    _, axes = plt.subplots(1, 1, sharex=True, sharey = True, figsize=figsize, gridspec_kw={'wspace': 0, 'bottom':0.15})
    g = sns.barplot(data = mean_metrics_df, x = mean_metrics_df.columns[0], y = mean_metrics_df.index, palette = palette, hue = mean_metrics_df.columns[0], dodge = False, ax = axes)
    g.set_xticklabels([])
    g.tick_params(labelsize=20)
    g.set(xticks=[])
    g.set(ylabel = None)
    g.set(xlabel = None)
    g.legend([],[], frameon=False)
    plt.savefig(os.path.join(save_dir, filename_save + '_mean_ranking.pdf'), bbox_inches = "tight")