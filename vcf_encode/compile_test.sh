#!/usr/bin/env bash

# Get the absolute path of the parent directory.
# In this case, that corresponds to this repositories root directory.
genosis_root=$(realpath ..)

singularity exec \
    --bind /scratch:/scratch \
    --bind ${genosis_root}:/genosis_root \
    --bind ${PWD}:/workspace \
    docker://mchowdh200/genosis:latest \
    g++ -std=c++11 \
        /genosis_root/cpp/src/interpolate_map.cpp \
        -I /genosis_root/cpp/include/ \
        -o /workspace/interpolate_map
