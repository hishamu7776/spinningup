
# train cartpole experiments
./cartpole.py --train-all --config-path './configs/cartpole.yaml'

# train pendulum experiments
./inverted_pendulum.py --train-all --config-path './configs/pendulum_stabilize.yaml'

# train satellite docking experiments
./docking.py --train-all --config-path './configs/docking.yaml'

# train aircraft rejoin experiments
./rejoin.py --train-all --config-path './configs/rejoin.yaml'

# train lava crossing experiments
./lava_crossing.py --train-all --config-path './configs/avoid_lava.yaml'

# train door-key experiments
./door_key.py --train-all --config-path './configs/cartpole.yaml'

# train robotic arm experiments
./reacher.py --train-all --config-path './configs/reacher.yaml'