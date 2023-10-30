# SEED: Primitive Skill-Based Robot Learning from Human Evaluative Feedback
This repository contains code to reproduce experiments in [SEED: Primitive Skill-Based Robot Learning from Human Evaluative Feedback](https://seediros23.github.io/).

## Installation
This repository is mainly built on top of [robosuite](https://robosuite.ai/) and [stable-baselines3](https://github.com/DLR-RM/stable-baselines3).
We use the forked version of the above libraries, where we implemented additional algorithms and features on top of them:
  - [robosuite](https://github.com/misoshiruseijin/robosuite)
  - [stable-baselines3](https://github.com/mj-hwang/stable-baselines3)

To begin, create a python/conda virtual environment, and install robosuite and stable-baselines3.
For the robosuite installation, follow the folling steps:
```sh
cd robosuite
pip3 install -r requirements.txt
pip3 install -r requirements-extra.txt
```

For the stable-baselines3 installation, follow the folling steps:
```sh
pip install -e
```

## Usage
The forked repository of [robosuite](https://robosuite.ai/) contains our primitive skill implementation (using operational space controller), as well as our custon environments.
We have two simple simulation environments: Reaching and Stacking.

The forked repository of [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) has implemention of our proposed algorithm, SEED, as well as other baselines (MAPLE, TAMER). 

Finally, to run the exepriments, run 
```sh
python train_seed_env.py
```

## Citing our work
If you use SEED for academic research, please kindly cite with the following BibTeX entry.

```
@article{hiranaka2023primitive,
  title={Primitive Skill-based Robot Learning from Human Evaluative Feedback},
  author={Hiranaka, Ayano and Hwang, Minjune and Lee, Sharon and Wang, Chen and Fei-Fei, Li and Wu, Jiajun and Zhang, Ruohan},
  journal={arXiv preprint arXiv:2307.15801},
  year={2023}
}
```

## Contact for questions or discussions
If you have any question or would like to discuss regarding the work, please feel free to contanct mjhwang@stanford.edu!!
