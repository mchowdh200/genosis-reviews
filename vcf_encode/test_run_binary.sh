#!/usr/bin/env bash

genosis_root=$(realpath ..)

singularity exec \
    --bind ${genosis_root}:/genosis_root \
    --bind /scratch:/scratch \
    --bind ${PWD}:/workspace \
    docker://mchowdh200/genosis:latest \
    ./interpolate_map "$@"
