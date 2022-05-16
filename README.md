# DRL-MTVCS
Additional materials for paper "[Multi-Task-Oriented Vehicular Crowdsensing: A Deep Learning Approach](https://ieeexplore.ieee.org/document/9155393)" accepted by IEEE INFOCOM 2020.

## :page_facing_up: Description
With the popularity of drones and driverless cars, vehicular crowdsensing (VCS) becomes increasingly widely-used by taking advantage of their high-precision sensors and durability in harsh environments. Since abrupt sensing tasks usually cannot be prepared beforehand, we need a generic control logic fit-for-use all tasks which are similar in nature, but different in their own settings like Point-of-Interest (PoI) distributions. The objectives include to simultaneously maximize the data collection amount, geographic fairness, and minimize the energy consumption of all vehicles for all tasks, which usually cannot be explicitly expressed in a closed-form equation, thus not tractable as an optimization problem. In this paper, we propose a deep reinforcement learning (DRL)-based centralized control, distributed execution framework for multi-task-oriented VCS, called "DRL-MTVCS". It includes an asynchronous architecture with spatiotemporal state information modeling, multi-task-oriented value estimates by adaptive normalization, and auxiliary vehicle action exploration by pixel control. We compare with three baselines, and results show that DRL-MTVCS outperforms all others in terms of energy efficiency when varying different numbers of tasks, vehicles, charging stations and sensing range.

## :wrench: Installation
1. Clone repo
    ```bash
    git clone https://github.com/BIT-MCS/DRL-MTVCS.git
    cd DRL-MTVCS
    ```
2. Install dependent packages
    ```sh
    conda create -n mcs python==3.8
    conda activate mcs
    pip install tensorflow-gpu==1.15
    pip install -r requirements.txt
    ```


## :computer: Training

Train our solution
```bash
python experiment.py --mode=train --logdir=./log
```
Use pupulation-based training (PBT) to tune the parameters [experimental]
```bash
python pbt_train.py
```

## :checkered_flag: Testing

Test with the trained models 

```sh
python experiment.py --mode=test --logdir=./log
```

Random test the env

```sh
python test_crazymcs.py
```

## :clap: Reference
- https://github.com/deepmind/scalable_agent


## :scroll: Acknowledgement

This paper was supported by National Natural Science
Foundation of China (No. 61772072).
<br>
Corresponding author: Chi Harold Liu.

## :e-mail: Contact

If you have any question, please email `3120215520@bit.edu.cn`.
