Machine LeaRning Assignment
================
Aimer G. Diaz

# 1. Background and Objectives

Classical work on plant immunity has emphasized transcriptional
induction of defense genes, largely because it has relied on
RNA-seq–based approaches. However, recent studies from the Xinnian Dong
lab and others have established that translational control constitutes a
central and independent regulatory layer, particularly during rapid
responses to pathogen attack \[[1](#ref-xiang2025translational)\].

A series of high-impact papers from the Dong lab has described several,
partly overlapping mechanisms that modulate translation of
immune-related mRNAs during Pattern-Triggered Immunity (PTI) and
Effector-Triggered Immunity (ETI). During PTI, global translation is
broadly repressed through mRNA decapping. Nevertheless, a subset of
defense mRNAs escapes this repression because they contain a purine-rich
"R-motif" in their untranslated regions. This R-motif functions as an
IRES-like element that recruits poly(A)-binding proteins (PABPs) and
enables cap-independent translation \[[2](#ref-xu2017global)\]
\[[3](#ref-wang2022pabp)\] and \[[4](#ref-xiang2023pervasive)\]

Taken together, these studies suggest that translation of immune-related
genes is governed by a complex regulatory code involving:

- PABP binding to purine-rich (R-) motifs

- Cap-independent translation initiation and IRES-like behavior

- Specific sequence motifs (AG-/purine-rich tracts) in untranslated
  regions

- Higher-order RNA structures (e.g. downstream RNA hairpins)

However, because these mechanisms have been described across multiple,
highly focused studies, it remains difficult to assemble a single,
unified model explaining which RNA features actually govern the
translational enhancement of individual immune-related genes upon
pathogen challenge. In particular, it is still unclear whether
translational activation is primarily dictated by the presence and
position of R-motifs and PABP-binding sites, or instead by RNA secondary
structures, such as predicted IRES-like elements or downstream hairpins.

## Objectives of this project

This machine-learning project aims to integrate Dong-lab datasets and
concepts into a unified, predictive framework for immune-gene
translational control. Specifically, using the most recent set of genes
described as carrying immune-responsive structural elements in their 5′
untranslated regions \[[4](#ref-xiang2023pervasive)\], we will evaluate
how well a model trained on these genes can classify immune-related
genes from the original R-motif–based dataset described in
\[[2](#ref-xu2017global)\]. Notably, this dataset comprises 4,727 genes,
representing approximately 17% of all protein-coding genes in
Arabidopsis. The breadth of this gene set suggests that it may, at least
in part, reflect the limited specificity of classical motif-based
prediction approaches, motivating the use of integrative and data-driven
modeling strategies.

## Variables

- *Arabidopsis* gene ID
- RNA hairpin governing translational Initiation (1/-1)
- R motif score
- A proportion (0 to 1)
- 5’UTR length
- Distance to mORF
- Average transcription in unchallenge conditions
- Average transcription in challenge conditions
- Average translation in unchallenge conditions
- Average translation in challenge conditions
- Immune Induction

# 2. Data Import and Cleaning

The data with the structural information of immune related genes
\[[4](#ref-xiang2023pervasive)\] can be found in the [TISnet github
repository](https://github.com/huangwenze/TISnet/blob/main/data/input_data/TIS_data.tsv),
and stored in the [Data
folder](Data/2023.TISNET.Pervasive_downstream_RNA_hairpins_dynamically_dictate_start-codon_selection.tsv),
the Extended figure 5 from \[[4](#ref-xiang2023pervasive)\] explain with
great detail what kind of information has this dataset:

<div style="text-align: center;">

<figure>
<img src="Figures/Tisnet_data.png" style="width: 100%;
height: auto"/>
<figcaption style="margin-top: 10px;">

The RNA secondary structures downstream of AUGs were predicted by
RNAfold constrained by SHAPE reactivities. TISnet predicted the
probability of initiating AUG by integrating the RNA primary sequence
and secondary structure information. AUGs with probability ≥ 0.9 are
defined as predicted initiating AUGs, and AUGs with probability \< 0.9
are defined as predicted noninitiating AUGs.

</figcaption>
</figure>

<a name="Tisnet_data.png"></a>

</div>

To select the genes that will be used in the following experiments a
series of bash commands were applied to simplify data selection

``` bash
cut -f 2,5,6  2023.TISNET.Pervasive_downstream_RNA_hairpins_dynamically_dictate_start-codon_selection.tsv | awk ' {gsub(/\..*/,"",$1); print $1,$2,$3 } ' | sort | uniq  | sort -k 2 | grep -v "\-1"  > TISNET_Shape_Pos.txt

 cut -f 2,5,6  2023.TISNET.Pervasive_downstream_RNA_hairpins_dynamically_dictate_start-codon_selection.tsv | awk ' {gsub(/\..*/,"",$1); print $1,$2,$3 } ' | sort | uniq  | sort -k 2 | grep  "\-1"  > TISNET_Shape_Neg.txt
 
grep -vf <( awk '{ print $1} ' TISNET_Shape_Pos.txt) TISNET_Shape_Neg.txt  > TISNET_Shape_TrueNeg.txt
 
cat TISNET_Shape_Pos.txt TISNET_Shape_TrueNeg.txt > TISNET_Shape.txt 
```

``` r
TISNET_Shape <- read.csv("Data/TISNET_Shape.txt",sep = " ", header = FALSE)
TISNET_Shape <- TISNET_Shape[,-2]

colnames(TISNET_Shape) <- c("GeneID","AUG_Hairpin")
table(TISNET_Shape$AUG_Hairpin)
```

    ## 
    ##   -1    1 
    ##  535 2857

Which means, 535 genes has RNA structures in the 5’UTR that do not
govern the start codon selection during an immune challenge, while 2857
of them does.

The additional data comes from the supplementary files of
\[[2](#ref-xu2017global)\], specially the [supplementary file
1](https://static-content.springer.com/esm/art%3A10.1038%2Fnature22371/MediaObjects/41586_2017_BFnature22371_MOESM2_ESM.xlsx)
and [supplementary file
5](https://static-content.springer.com/esm/art%3A10.1038%2Fnature22371/MediaObjects/41586_2017_BFnature22371_MOESM6_ESM.xlsx)

``` r
Transcription_Translation <- read.csv("Data/2017.SUP1.Global_translational_reprogramming_RNARibo.csv", header = T)
colnames(Transcription_Translation)<- gsub("geneID","GeneID",colnames(Transcription_Translation))
Features <- read.csv("Data/2017.SUP5.Global_translational_reprogramming_RMOTIF.csv", header = T)

AllInfo <- merge(Transcription_Translation, Features , by.x =  "GeneID", all.x = T)

TISNET_Rmotif <- merge(TISNET_Shape, AllInfo,  by.x =  "GeneID")
table(TISNET_Rmotif$AUG_Hairpin)
```

    ## 
    ##   -1    1 
    ##  286 1507

``` r
table(is.na(TISNET_Rmotif$score))
```

    ## 
    ## FALSE  TRUE 
    ##   470  1323

``` r
table(is.na(TISNET_Rmotif$score),TISNET_Rmotif$AUG_Hairpin)
```

    ##        
    ##           -1    1
    ##   FALSE   73  397
    ##   TRUE   213 1110

The data integration reduced severly the number of genes to test, from a
total of 3392 to 1793, from which only 470 has R motif prediction,
distributed like 73 without AUG initiating RNA structure and 397 AUG
initiating structure.

# References

<div id="refs" class="references csl-bib-body">

<div id="ref-xiang2025translational" class="csl-entry">

1\. Xiang Y, Dong X. Translational regulation of plant stress responses:
Mechanisms, pathways, and applications in bioengineering. Annual Review
of Phytopathology. 2025;63.

</div>

<div id="ref-xu2017global" class="csl-entry">

2\. Xu G, Greene GH, Yoo H, Liu L, Marqués J, Motley J, et al. Global
translational reprogramming is a fundamental layer of immune regulation
in plants. Nature. 2017;545:487–90.

</div>

<div id="ref-wang2022pabp" class="csl-entry">

3\. Wang J, Zhang X, Greene GH, Xu G, Dong X. PABP/purine-rich motif as
an initiation module for cap-independent translation in
pattern-triggered immunity. Cell. 2022;185:3186–200.

</div>

<div id="ref-xiang2023pervasive" class="csl-entry">

4\. Xiang Y, Huang W, Tan L, Chen T, He Y, Irving PS, et al. Pervasive
downstream RNA hairpins dynamically dictate start-codon selection.
Nature. 2023;621:423–30.

</div>

</div>
