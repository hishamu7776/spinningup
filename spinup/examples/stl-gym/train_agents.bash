
# train cartpole experiments
python cartpole.py --train-all --config-path './configs/cartpole.yaml'

# train pendulum experiments
python inverted_pendulum.py --train-all --config-path './configs/pendulum_stabilize.yaml'

# train satellite docking experiments
python docking.py --train-all --config-path './configs/docking.yaml'

# train aircraft rejoin experiments
python rejoin.py --train-all --config-path './configs/rejoin.yaml'

# train lava crossing experiments
python lava_crossing.py --train-all --config-path './configs/avoid_lava.yaml'

# train door-key experiments
python door_key.py --train-all --config-path './configs/cartpole.yaml'

# train robotic arm experiments
python reacher.py --train-all --config-path './configs/reacher.yaml'