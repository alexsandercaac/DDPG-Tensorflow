DDPG Tensorflow
==============================

This repository provides an updated version of the DDPG implementation published in the official Tensorflow keras [website](https://keras.io/examples/rl/ddpg_pendulum/) that works with the new [Gymnasium](https://gymnasium.farama.org/content/gym_compatibility/) API (formerly known as `gym`). In addition to updating the original Jupyter notebook, different Python modules are created for construction of a [DVC](https://dvc.org/) pipeline.

The DDPG Algorithm 🤖
------------------

The DDPG algorithm is an actor-critic method that uses two neural networks to approximate the optimal policy and Q-function. The policy network is updated via the policy gradient theorem while the Q-function is updated via the Bellman equation. The two networks are trained with a common replay buffer.

The DDPG algorithm can be applied to continuous action spaces. The [Pendulum environment](https://gymnasium.farama.org/environments/classic_control/pendulum/) in Gymnasium is a good example of a continuous action space. The Pendulum environment is a two-dimensional state space with three continuous actions. The state space consists of the sine and cosine of the angle and the angular velocity. The action space is the torque applied to the pendulum.

For more information on the DDPG algorithm, see the [original paper](https://arxiv.org/abs/1509.02971) or the Spinning Up [docs page](https://spinningup.openai.com/en/latest/algorithms/ddpg.html).

Repository Structure 📁
--------------------

In the `notebooks` folder, the adapted original Jupyter notebook is provided --- the adaptations were made with the goal of updating the code to the newest versions of the main python packages used.

The `src` folder contains the Python modules that are used to construct the DVC pipeline. Some of the main modules in the `src` folder include:

- `ddpg.py`: contains the DDPG class that is used to train the agent. The file is located in the `src/agents` folder.
- `action_noise.py`: contains different classes that implement various noise generating processes that are used to add noise to the actions of the agent. The file is located in the `src/utils` folder.
- `buffer.py`: contains the replay buffer class that is used to store the agent's experience. The file is located in the `src/utils` folder.
- `anns.py`: contains the neural network classes that are used to construct the policy and Q-function networks. The file is located in the `src/utils` folder.

Instructions 📄
--------------------

#### DVC

The experiments in this repository were tracked using a tool called [DVC](https://github.com/iterative/dvc), which is a command line tool for the development of reproducible machine learning projects.

With DVC, data and artifacts from experiments can be tracked both locally and remotely. For this repository, the cache with the full history of development is made available for public use with read-only access. This remote is already configured by default in the `.dvc/config` file, and will allow you to download all of the models and artifacts generated in the development of this project. **Note:** you may be ask to authenticate your identity with gdrive, and give DVC permissions --- since the data is public, any account will give you access to the repo. 

If you would like access to a remote repository with writing access, however, you will need to create your own. Instructions for that can be found in the [DVC website](https://dvc.org/doc/user-guide/data-management/remote-storage).


