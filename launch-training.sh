#!/usr/bin/env bash
#SBATCH --partition=long
#SBATCH --job-name=genosis-train
#SBATCH --output=logs/genosis-train.out
#SBATCH --error=logs/genosis-train.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=48:00:00

snakemake -s training_pipeline.smk \
    --slurm-logdir logs \
    --workflow-profile profiles/slurm \
    --jobs 32 \
    --cores 32

