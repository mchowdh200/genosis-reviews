import os
import random
import sys
from glob import glob
from types import SimpleNamespace

from workflow_utils.training_pipeline_setup import (
    get_segments,
    train_val_test_split,
    get_wandb_api_key,
)


configfile: "conf/exp_config.yaml"


# ==============================================================================
# Workflow setup
# ==============================================================================
config = SimpleNamespace(**config)

segments = get_segments(
    config.segments_dir,
    config.seg_offset,
    config.gt_ext,
    config.file_segment_delimiter,
)

train_segments, val_segments, test_segments = train_val_test_split(
    config.random_seed,
    segments,
)

# NOTE NEED to have WANDB_API_KEY set in the environment.
# either set beforehand, or have your bashrc/bash_profile source it.
# Just DO NOT have it anywhere visible in checked in code.
wandb_api_key = get_wandb_api_key()


# ==============================================================================
# Rules
# ==============================================================================
rule All:
    input:
        # trained model
        directory(f"{config.outdir}/{config.model_prefix}.checkpoints"),
        # train memmaps
        f"{config.outdir}/training_set/P1.mmap",
        f"{config.outdir}/training_set/P2.mmap",
        # val memmaps
        f"{config.outdir}/validation_set/P1.mmap",
        f"{config.outdir}/validation_set/P2.mmap",
        # train/test/split segement numbers
        f"{config.outdir}/train.segments",
        f"{config.outdir}/val.segments",
        f"{config.outdir}/test.segments",
        # training set
        expand(f"{config.outdir}/training_set/D.txt", segment=train_segments),
        # validation set
        expand(f"{config.outdir}/validation_set/D.txt", segment=val_segments),
        # stats
        # f"{config.outdir}/bin_sampled_distribution.pdf",
        # f"{config.outdir}/distribution.pdf",


rule SetupLogDir:
    """
    Create the log directory for the run
    """
    output:
        directory("logs"),
    shell:
        f"""
        mkdir -p logs
        """


rule SampleSubpops:
    input:
        config.sample_table,
    output:
        f"{config.outdir}/subpops/pairings.{{segment}}.txt",
    shell:
        f"""
        python exploration/sample_subpops.py \
            --sample_table {{input}} \
            --samples_list {config.samples_list} \
            --output {{output}} \
            --N {config.num_pairings}
        """


rule MakeGTMemmap:
    """
    For each segment, make a numpy memmap of the genotype matrix
    """
    input:
        f"{config.segments_dir}/{config.segment_prefix}.{{segment}}.{config.gt_ext}",
    output:
        f"{config.outdir}/gt_mmap/segment.{{segment}}.mmap",
    container:
        "docker://mchowdh200/genosis:latest"
    shell:
        f"""
        python exploration/make_gt_mmap.py \
            --gts {{input}} \
            --output {{output}} \
        """


rule ComputeDistances:
    """
    For each sample, get the distances of all the samples paired with it.
    Since the paired samples don't have haplotype numbers, randomly choose one
    """
    input:
        memmap=rules.MakeGTMemmap.output,
        pairings=rules.SampleSubpops.output,
    output:
        f"{config.outdir}/distances/distances.{{segment}}.txt",
    container:
        "docker://mchowdh200/genosis:latest"
    shell:
        f"""
        python exploration/compute_distances.py \
            --memmap {{input.memmap}} \
            --pairings {{input.pairings}} \
            --output {{output}} \
            --samples_list {config.samples_list}
        """


rule PlotDistribution:
    """
  Plot the distribution of distances
  """
    input:
        rules.ComputeDistances.output,
    output:
        temp(f"{config.outdir}/plots/distribution.{{segment}}.png"),
    shell:
        f"""
    python exploration/plot_distribution.py \
      --segment {{wildcards.segment}} \
      --distances {{input}} \
      --output {{output}}
    """


rule CatImagesPDF:
    """
  Concatenate all the images into a single pdf
  """
    input:
        expand(rules.PlotDistribution.output, segment=segments),
    output:
        f"{config.outdir}/distribution.pdf",
    shell:
        f"""
    convert {{input}} {{output}}
    """


rule BinSampling:
    """
    For each segment, divide the space of possible distances into bins and
    sample from each bin to get a representative sample of distances.
    """
    input:
        memmap=rules.MakeGTMemmap.output,
        distances=rules.ComputeDistances.output,
    output:
        f"{config.outdir}/bin_sampling/segment.{{segment}}.txt",
    container:
        "docker://mchowdh200/genosis:latest"
    shell:
        f"""
        python exploration/bin_sampling.py \
            --distances {{input.distances}} \
            --output {{output}} \
            --num_bins 20 \
            --max_samples 500
        """


rule PlotResampledDistribution:
    """
    Plot the distribution of distances from the resampled distances
    """
    input:
        rules.BinSampling.output,
    output:
        temp(f"{config.outdir}/resampled_plots/distribution.{{segment}}.png"),
    shell:
        f"""
        python exploration/plot_distribution.py \
            --segment {{wildcards.segment}} \
            --distances {{input}} \
            --output {{output}}
        """


rule CatResampledImagesPDF:
    """
    Concatenate all the images into a single pdf
    """
    input:
        expand(rules.PlotResampledDistribution.output, segment=segments),
    output:
        f"{config.outdir}/bin_sampled_distribution.pdf",
    shell:
        f"""
        convert {{input}} {{output}}
        """


rule WriteTrainTestSplit:
    """
    Write the train and test segment numbers to a file
    """
    output:
        train=f"{config.outdir}/train.segments",
        val=f"{config.outdir}/val.segments",
        test=f"{config.outdir}/test.segments",
    run:
        with open(output.train, "w") as f:
            for segment in sorted(map(int, train_segments)):
                f.write(f"{segment}\n")
        with open(output.val, "w") as f:
            for segment in sorted(map(int, val_segments)):
                f.write(f"{segment}\n")
        with open(output.test, "w") as f:
            for segment in sorted(map(int, test_segments)):
                f.write(f"{segment}\n")


rule MakeTrainingSet:
    """
    From the resampled distances, make a training set of pairs
    of samples, over all segments.
    """
    input:
        pos_files=expand(
            f"{config.segments_dir}/{config.segment_prefix}.{{segment}}.{config.pos_ext}",
            segment=train_segments,
        ),
        distances=expand(rules.BinSampling.output, segment=train_segments),
    output:
        P1=temp(f"{config.outdir}/training_set/P1.txt"),
        P2=temp(f"{config.outdir}/training_set/P2.txt"),
        D=f"{config.outdir}/training_set/D.txt",
    params:
        subtract_segment_from_pos=(
            "--subtract_segment_from_pos" if config.subtract_segment_from_pos else ""
        ),
    shell:
        f"""
        python exploration/make_dataset.py \
            {{params.subtract_segment_from_pos}} \
            --pos_files {{input.pos_files}} \
            --distance_files {{input.distances}} \
            --P1 {{output.P1}} \
            --P2 {{output.P2}} \
            --D {{output.D}} \
        """


rule MakeValidationSet:
    """
    From the resampled distances, make a validation set of pairs
    of samples, over all segments.
    """
    input:
        pos_files=expand(
            f"{config.segments_dir}/{config.segment_prefix}.{{segment}}.{config.pos_ext}",
            segment=val_segments,
        ),
        distances=expand(rules.BinSampling.output, segment=val_segments),
    output:
        P1=temp(f"{config.outdir}/validation_set/P1.txt"),
        P2=temp(f"{config.outdir}/validation_set/P2.txt"),
        D=f"{config.outdir}/validation_set/D.txt",
    params:
        subtract_segment_from_pos=(
            "--subtract_segment_from_pos" if config.subtract_segment_from_pos else ""
        ),
    shell:
        f"""
        python exploration/make_dataset.py \
            {{params.subtract_segment_from_pos}} \
            --pos_files {{input.pos_files}} \
            --distance_files {{input.distances}} \
            --P1 {{output.P1}} \
            --P2 {{output.P2}} \
            --D {{output.D}} \
        """


rule MakeTrainMmaps:
    """
    Make memmaps for the training set position vectors
    """
    input:
        P1=rules.MakeTrainingSet.output.P1,
        P2=rules.MakeTrainingSet.output.P2,
    output:
        P1=directory(f"{config.outdir}/training_set/P1.mmap"),
        P2=directory(f"{config.outdir}/training_set/P2.mmap"),
    container:
        "docker://mchowdh200/genosis:latest"
    shell:
        f"""
        python exploration/generate_mmaps.py \
            --inP1 {{input.P1}} \
            --inP2 {{input.P2}} \
            --outP1 {{output.P1}} \
            --outP2 {{output.P2}}
        """


rule MakeValMmaps:
    """
    Make memmaps for the validation set position vectors
    """
    input:
        P1=rules.MakeValidationSet.output.P1,
        P2=rules.MakeValidationSet.output.P2,
    output:
        P1=directory(f"{config.outdir}/validation_set/P1.mmap"),
        P2=directory(f"{config.outdir}/validation_set/P2.mmap"),
    container:
        "docker://mchowdh200/genosis:latest"
    shell:
        f"""
        python exploration/generate_mmaps.py \
        --inP1 {{input.P1}} \
        --inP2 {{input.P2}} \
        --outP1 {{output.P1}} \
        --outP2 {{output.P2}}
        """


rule TrainModel:
    input:
        train_segments=f"{config.outdir}/train.segments",
        val_segments=f"{config.outdir}/val.segments",
        test_segments=f"{config.outdir}/test.segments",
        P1_train=rules.MakeTrainMmaps.output.P1,
        P2_train=rules.MakeTrainMmaps.output.P2,
        P1_val=rules.MakeValMmaps.output.P1,
        P2_val=rules.MakeValMmaps.output.P2,
        D_train=rules.MakeTrainingSet.output.D,
        D_val=rules.MakeValidationSet.output.D,
    output:
        model_checkpoints=directory(
            f"{config.outdir}/{config.model_prefix}.checkpoints"
        ),
    threads: workflow.cores
    container:
        "docker://mchowdh200/genosis:latest"
    shell:
        f"""
        WANDB_API_KEY={wandb_api_key} python train_model.py \
            --outdir {config.outdir} \
            --model_prefix {config.model_prefix} \
            --P1_train {{input.P1_train}} \
            --P2_train {{input.P2_train}} \
            --P1_val {{input.P1_val}} \
            --P2_val {{input.P2_val}} \
            --D_train {{input.D_train}} \
            --D_val {{input.D_val}} \
            --model_config {config.model_config}
        """
