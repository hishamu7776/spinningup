# STLGym examples


## Installation Instructions
Assuming `conda` is installed, run the following from the desired location for all necessary repos to be stored

```bash
conda create --name stlgym python=3.6
conda activate stlgym
git clone https://github.com/nphamilton/spinningup.git
cd spinningup
pip install -e .
cd ..
git clone --recursive https://github.com/nphamilton/stl-gym.git
cd stl-gym/rtmt
pip install -e .
cd ..
pip install -e .
cd ..
```

If you also want to run the `reacher.py` example, `mujoco-py` needs to be downloaded, installed, and setup. Follow the directions at [https://github.com/openai/mujoco-py#install-mujoco](https://github.com/openai/mujoco-py#install-mujoco) to get the correct binary files saved to the correct directory. Then run the following:

```bash
pip install -U 'mujoco-py<2.2,>=2.1'
sudo apt-get install libglew-dev
sudo apt-get install patchelf
```