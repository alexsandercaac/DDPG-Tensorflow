DDPG Tensorflow
==============================

This repository provides an updated version of the DDPG implementation published in the official keras [website](https://keras.io/examples/rl/ddpg_pendulum/) that works with the new [Gymnasium](https://gymnasium.farama.org/content/gym_compatibility/) API (formerly known as `gym`).

An extended implementation of the DDPG algorithm is also provided as a python module.

Project Organization
-----------
***

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    └── src                <- Source code for use in this project.
       │
       └── agents  <- Scripts to create RL agents
           └── ddpg.py



--------

<p><small>Project adapted from the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
