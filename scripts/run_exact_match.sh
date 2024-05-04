#!/bin/bash

dir="/mnt/shareddata/yiyan/grid_search/gs_050224_more_hyperparams"
cd "$dir"
for subdir in *; do
    echo "#########################################################"
    echo "Working on $subdir..."
    inf_dir="$dir/$subdir/inference"
    python /home/yiyan_hao/breast/scripts/compute_overall.py --output_directory "$inf_dir"
done
