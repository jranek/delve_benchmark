library("SingleCellExperiment")
library("slingshot")
run_slingshot <- function(embedding, labels, root_cluster){
    #### performs trajectory inference with Slingshot: https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-018-4772-0
    parameters <- list()
    parameters$shrink <- 1L
    parameters$reweight <- TRUE
    parameters$reassign <- TRUE
    parameters$thresh <- 0.001
    parameters$maxit <- 10L
    parameters$stretch <- 2L
    parameters$smoother <- "smooth.spline"
    parameters$shrink.method <-"cosine"

    tryCatch(
        expr = {
            sds <- slingshot::slingshot(embedding, labels, start.clus = root_cluster)
            # collect milestone network
            lineages <- slingLineages(sds)
            pseudotime <- slingPseudotime(sds)

            output <- list()
            output$lineages <- lineages
            output$pseudotime <- pseudotime

            return(output)
},
            
        error = function(e){
            print(e)
            output <- list()
            output$lineages <- NaN
            output$pseudotime <- NaN
            return(output)
        }
    )
}