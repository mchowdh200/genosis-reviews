#!/usr/bin/env bash
#SBATCH --partition=long
#SBATCH --job-name=genosis-train
#SBATCH --output=logs/no-afr-train.out
#SBATCH --error=logs/no-afr-train.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=72:00:00

snakemake -s training_pipeline.smk \
    --configfile ./conf/no_afr_training_conf.yaml \
    --slurm-logdir logs \
    --workflow-profile profiles/slurm \
    --jobs 1 \
    --cores 32 \
    --rerun-triggers mtime
