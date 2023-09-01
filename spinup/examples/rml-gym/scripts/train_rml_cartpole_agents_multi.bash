#!/bin/bash

# Set the random seeds
random_seeds=(1630 2241 2320 3281 5640 9348)

# For each random seed, execute the python samples
for rs in "${random_seeds[@]}"
do
  echo "Running python samples with random seed $rs"
  #python test_rml_cartpole_multi_algorithm.py --ppo --random-seed $rs
  python test_rml_cartpole_multi_algorithm.py --trpo --random-seed $rs
  #python test_rml_cartpole_multi_algorithm.py --td3 --random-seed $rs
  #python test_rml_cartpole_multi_algorithm.py --vpg --random-seed $rs
done
