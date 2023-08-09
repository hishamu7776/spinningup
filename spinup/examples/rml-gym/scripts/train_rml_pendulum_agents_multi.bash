#!/bin/bash

# Set the random seeds
random_seeds=(1630 2241 2320 2990 3281 4930 5640 8005 9348 9462)

# For each random seed, execute the python samples
for rs in "${random_seeds[@]}"
do
  echo "Running python samples with random seed $rs"
  python test_rml_pendulum_multi_algorithm.py --ppo --random-seed $rs
  python test_rml_pendulum_multi_algorithm.py --sac --random-seed $rs
  python test_rml_pendulum_multi_algorithm.py --td3 --random-seed $rs
  python test_rml_pendulum_multi_algorithm.py --vpg --random-seed $rs
done
