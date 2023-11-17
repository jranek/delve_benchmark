library("SymSim")
symsim_generate <- function(ncells_total, ngenes, ce, Sigma, random_state){
    #### simulates single-cell RNA sequencing data with SymSim: https://www.nature.com/articles/s41467-019-10500-w
    true_counts_res <- SimulateTrueCounts(ncells_total=ncells_total, ngenes=ngenes, nevf = 0.015*ngenes, n_de_evf = 0.01*ngenes, evf_type="continuous", vary="s", Sigma=Sigma, phyla=Phyla3(), randseed=random_state)

    data(gene_len_pool)
    gene_len <- sample(gene_len_pool, ngenes, replace = FALSE)
    observed_counts <- True2ObservedCounts(true_counts=true_counts_res[[1]], meta_cell=true_counts_res[[3]], protocol="UMI", alpha_mean=ce, alpha_sd=0.02, gene_len=gene_len, depth_mean=5e5, depth_sd=3e4)
    
    TrajInfo <- getTrajectoryGenes(observed_counts$cell_meta)

    output = list()
    output$X = t(observed_counts[[1]])
    output$groups = TrajInfo["branch"]
    output$pseudotime = TrajInfo["pseudotime"]
    return(output)
}