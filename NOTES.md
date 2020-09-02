### Install in an editable mode:
```bash
git clone https://github.com/AthinkingNeal/SC2-ACME.git
cd acme
conda create --name acme python=3.7
conda activate acme
pip install -e . # install dm-acme
pip install -e .[reverb]
pip install -e .[tf]
pip install -e .[envs]
pip install -e .[jax]
pip install .[testing]
pip install gym[atari]
```

Compatible CUDA Version: 10.2