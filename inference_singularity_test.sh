#!/usr/bin/env bash
#SBATCH --job-name=inference_singularity_test
#SBATCH --partition=nvidia-a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=output-logs.txt
#SBATCH --error=error-logs.txt

# absolute path to the directory containing input data
BASE_DIR='/scratch/Shares/layer/projects/gt_similarity_search'

# filenames of test positional encodings (space separated)
TEST_SEGMENT='segment.156.pos'

singularity exec \
    --nv \
    --bind /scratch:/scratch \
    --bind ${PWD}:/workspace \
    /scratch/Shares/layer/containers/genosis.sif \
    python encode_samples.py \
    --output "test.txt" \
    --files "${BASE_DIR}/segments/${TEST_SEGMENT}" \
    --batch-size 512 \
    --gpu \
    --num-workers 16 \
    --model-config conf/conv1d_config.yaml # PASS IN MODEL CONFIG WHEN WE OMIT THE ENCODER ARGUMENT
    # --encoder last.ckpt DONT PASS IN ENCODER ARGUMENT WHEN WE PASS IN MODEL CONFIG
