
options(
    download.file.method = "wget"
)
options(warn = -1)

library(clusterProfiler)

sys_argv <- commandArgs(T)

genes = readLines(sys_argv[1])
outfile = sys_argv[2]
# "SYMBOL", "Gg", "gga"
inType = sys_argv[3]
species = sys_argv[4]
organism = sys_argv[5]


out_res <- function(eg, inType, species, outfile, enrichtype){
    out_gene <- c()
    out_name <- c()
    for (each_termindex in seq(length(eg@result$ID))){
        out_name <- c(
            out_name,
            paste(
                eg@result$ID[each_termindex],
                eg@result$Description[each_termindex],
                sep="\t"
            )
        )
        tmp_gene <- clusterProfiler::bitr(
            geneID = strsplit(eg@result$geneID,"/")[[each_termindex]],
            fromType = "ENTREZID",
            toType = inType,
            OrgDb = paste0("org.", species, ".eg.db")
        )[,2]
        out_gene <- c(out_gene, paste(tmp_gene,collapse=","))
    }
    names(out_gene) <- out_name
    # eg@result$ID           eg@result$Description  eg@result$GeneRatio    eg@result$BgRatio      eg@result$pvalue       eg@result$p.adjust     eg@result$qvalue       eg@result$geneID       eg@result$Count
    out_matrix <- cbind(
        eg@result$pvalue,
        eg@result$GeneRatio,
        eg@result$p.adjust,
        eg@result$qvalue,
        out_gene
    )
    rownames(out_matrix) <- out_name
    colnames(out_matrix) <- c("pvalue", "GeneRatio", "p.adjust","qvalue","genes")
    write.table(
        out_matrix,
        file = paste0(outfile, ".", enrichtype, "Terms.txt"),
        col.names = T,
        row.names = T,
        sep = "\t",
        quote = F
    )
}

print("converting...")
gene.df <- bitr(
    geneID = genes,
    fromType = inType,
    toType = "ENTREZID",
    OrgDb = paste0("org.", species, ".eg.db")
)

tryCatch({
        print("GO...")
        eg <- enrichGO(
            gene = gene.df$ENTREZID,
            OrgDb = paste0("org.",species,".eg.db"),
            ont = "BP",
            pAdjustMethod = "BH",
            pvalueCutoff = 0.05,
            qvalueCutoff = 0.2,
            minGSSize = 2
        )
        # eg <- simplify(
        #     eg,
        #     cutoff=0.7,
        #     by="p.adjust",
        #     select_fun=min
        # )
        print("GO table...")
        out_res(eg, inType, species, outfile, "GO")

        print("GO plot...")
        pdf(
            paste0(outfile, ".GOTerms.pdf"),
            width = 10,
            height = 5
        )
        dotplot(
            eg,
            showCategory = 12,
            font.size = 10
        )
    },
    error=function(e){
        print("GO error")
    },
    finally=function(e){
    }
)


tryCatch({
        print("KEGG...")
        eg2 <- enrichKEGG(
            gene = gene.df$ENTREZID,
            organism = organism,
            keyType = "kegg",
            pAdjustMethod = "BH",
            pvalueCutoff = 0.05,
            qvalueCutoff = 0.2
        )

        print("KEGG table...")
        out_res(eg2, inType, species, outfile, "KEGG")

        print("KEGG plot...")
        pdf(
            paste0(outfile, ".KEGGTerms.pdf"),
            width = 10,
            height = 5
        )
        dotplot(
            eg2,
            showCategory = 12,
            font.size = 10
        )
    },
    error=function(e){
        print("KEGG error")
    }
)

print("done")