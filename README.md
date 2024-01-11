# MCAgeneKO
This repository covers the modeling, data analysis, and visualization techniques for predicting gene knockouts in malaria, aiding in the identification of potential candidates for drug discovery.

## Overview
Our initiative aims to promote drug discovery in the context of malaria parasite research by combining machine and deep learning techniques to decipher gene functions. Specifically, we used prediction algorithms to conduct a genome-wide search for critical genes required by malaria parasites. These models were carefully trained using data from Plasmodium berghei knockout mutant growth rates. We added gene and protein-centric features to improve the model's prediction capabilities. Our systematic efforts resulted in a noteworthy accuracy rate of 70% in predicting cases where specific genes are considered essential based only on gene-related traits.

## Prerequisites

Users need to install before using the Snakemake workflow.

- Python (>=3.7)

## Data collection and preprocessing 

Data for phenotypes were gathered from several research that examined genetic knockouts in Plasmodium berghei at various life cycle stages. As previously stated, a comprehensive genome-scale knockout investigation [Blood screen](https://pubmed.ncbi.nlm.nih.gov/28708996/) was conducted to better understand gene involvement throughout the blood stage. Because continuous drug selection for knockouts is only possible during the parasite's asexual blood phases, this strategy aids in the research of genes important in the parasite's later stages post-blood. Following research focused on the liver stage [Liver sceren](https://pubmed.ncbi.nlm.nih.gov/31730853/), while another bar-seq screen investigation found genes important for male and female gametocyte development. In addition, as stated in [Fetility screen](https://www.biorxiv.org/content/10.1101/2023.12.25.572958v1.abstract), a screening was performed to identify genes associated with male and female fertility. The phenotypic web interface compiles all of these screening datasets for easy access.

Single-cell gene expression data originate from the Malaria Cell Atlas ([MCA](https://www.malariacellatlas.org/)) and bulk-RNA gene expression data were collected from multiple studies, including [Study1](https://pubmed.ncbi.nlm.nih.gov/30177743/) and [Study2](https://pubmed.ncbi.nlm.nih.gov/28081440/). Additionally, bulk-RNA gene expression data are from diffrent life cycle stages of malaria parasites.

The preprocessing scripts can be found in the data_preprocessing directory within the code folder. To obtain all the necessary data, execute the following command:
~~~
python datautil.py
~~~

Furthermore, I've provided comprehensive data for training, validation, and testing encapsulated in a pickle file named "data_output.pkl", which resides in the data folder.

## Modeling

Users need to install before using the Snakemake workflow.

- Python (>=3.7)
- Snakemake (7.32.4)

## Installation

Install Snakemake using pip.
~~~
pip install snakemake
~~~
## Concepts and Descriptions
In this section, we will explain the terminology and concepts that were employed in the calculation of male and female fertility.

### Relative abundance
To calculate the abundance of mutants in each sample, we averaged the forward and reverse reads. Subsequently, we computed the abundance for each pool. For example, for a pool of 3 mutants/genes, the count matrix can be generated as shown in the following table.

| Genes | Sample  1 | Sample  2 |
|----------|----------|----------|
| Gene1 | 10.5 | 11.5 |
| Gene2 | 30 | 10.5 |
| Gene3|  20.5| 5.5 |



We determine the relative abundance by dividing the abundance of a specific feature by the total abundance in the sample. The resulting relative abundance table is as
follows:

| Genes | Sample  1 | Sample  2 |
|----------|----------|----------|
| Gene1 | 10.5/60 | 11.5/31 |
| Gene2 | 30/60 | 10/31 |
| Gene3|  20.5/60| 5.5/31 |

### Fertility

#### Step 1: Compute mean and variance for each mutant
On day 0 and day 13, we utilized relative abundance to calculate the mean and variance for each set of PCR duplicates. Subsequently, we applied a logarithmic transformation to the mean values (log2(mean)) and computed the relative variability represented by the coefficient of variation squared (CV^2).

#### Step 2: Inverse-variance weighting   
Inverse-variance weighting is a statistical technique that combines multiple random variables to reduce the variance of the weighted average. It is particularly useful in our analysis where we calculate the change in barcode abundance between day0 and day13. This change is computed as the difference between the logarithms of the mean abundance at day13 and day0. Additionally, we consider the variance of the data, which is propagated as the sum of relative variances at day0 and day13. This allows us to effectively assess changes in barcode abundance while accounting for the variability in the data.

#### Step 3: Normalized by spike-in controls
In our study, we included spike-in controls to normalize the change in barcode abundance. This normalized change in barcode abundance, which we refer as "fertility".

For each pool, we computed the fertility of mutants. Mutants were categorized as `Reduced` if their fertility, plus two times the standard deviation, fell below a certain cutoff value; otherwise, they were categorized as `Not Reduced`. Since spike-in controls were included in all pools, we employed inverse-variance weighting to obtain a consolidated measure of fertility and the associated error for this variable. This approach allowed us to effectively combine data from multiple pools while considering variations introduced by the spike-in controls.

## Usage

### Convert Fastq to count matrix
To convert paired forward and reverse reads from BARseq experiments into a count matrix, we employ the following command.
~~~
snakemake --use-conda --conda-frontend conda raw_barseq  --cores 1 -F --config input_file=barseq-raw/testdata/sequence barcode_file=barcode_to_gene_210920.csv output_dir=barseq-raw/testRes -p  
~~~
The process requires two key inputs: a directory (e.g., `barseq-raw/testdata/sequence`) where all the fastq files for the samples are stored and a CSV file (e.g., `barcode_to_gene_210920.csv`) containing barcode information for each gene or mutant. The resulting output is directed to another directory (e.g., `barseq-raw/testRes`), where both the mutant count matrix file and a log file are generated.

To identify and remove mutants with zero counts in all samples, use the following command. This will generate the file `removed_zeros_barcode_counts_table.csv`
in the `barseq-raw/testRes` directory.
~~~
snakemake --use-conda --conda-frontend conda remove_zeros  --cores 1 -F --config output_dir=barseq-raw/testRes -p
~~~




### Combine fertility screen data

In the Barseq experiment, we analyzed over 1200 mutants distributed across seven pools labeled as Pool1 to Pool7. Notably, Pool5 and Pool7 were studied twice. To consolidate the data from all these pools (Pool1 to Pool7), you can use the following command.
~~~
snakemake --use-conda --conda-frontend conda combine_pools  --cores 1 -F
~~~
This command/script will calculate male and female fertility, their variances, and male and female phenotypes (e.g., Reduced, Not reduced).

### Motility screen data
We conducted a BARseq experiment, focusing on the pool with the most strongest male fertility phenotypes. Subsequently, we collected barcodes from purified microgametes as part of a motility screen.

~~~
snakemake --use-conda --conda-frontend conda motility_screen  --cores 1 -F
~~~
This command/script will calculate motility rate, their variances, and motility phenotype (e.g., Reduced, Not reduced).

### Generate Visualizations
To visualize ranked male fertility and female fertility, as well as create a scatter plot of male vs. female fertility, you can utilize the following command.
~~~
snakemake --use-conda --conda-frontend conda plot_fertility  --cores 1 -F
~~~

We performed an enrichment analysis using the Malaria Parasite Metabolic Pathways (MPMP) for male and female-only fertility phenotypes. For this use following command.

~~~
snakemake --use-conda --conda-frontend conda mpmp_enrichment  --cores 1 -F
~~~

We visualize the enrichment of top-ranked pathways (male/female) using violin plots with the following commands.

~~~
snakemake --use-conda --conda-frontend conda mpmp_violin  --cores 1 -F
~~~

To investigate the relationship between fertility phenotypes and gene expression, we visualized genes associated with male and female-only fertility phenotypes within the Malaria Cell Atlas (mca) gene expression-based clusters.

~~~
snakemake --use-conda --conda-frontend conda mca_gene_plot  --cores 1 -F
~~~

To filter out noisy data, we generated a plot of relative abundance against relative error. Our analysis revealed that data with a relative abundance below the cutoff value of log2(-12) exhibited unacceptably high errors, making it unsuitable for further analysis.

~~~
snakemake --use-conda --conda-frontend conda error_plot  --cores 1 -F
~~~
