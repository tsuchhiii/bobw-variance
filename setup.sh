#!/usr/bin/env bash

python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# for linux server
# pip install numpy pandas matplotlib seaborn tqdm
# pip install torch torchvision torchaudio

# Python
export PYTHONIOENCODING=utf8
export PYTHONENCODING=utf8
# export PYTHONPATH=.
export PYTHONPATH="."

# make empty folders
# mkdir -p results/fig
# mkdir -p results/fig-linear-bandits

mkdir -p results/mab/fig
# mkdir -p results/pe-mab/fig
# mkdir -p results/linear/fig


# setup for fixed-budget project
# todo: actually we need to get trained policy data and specify it
#  like ./bandit/fixed_budget_pjt/checkpoint/3_Ber/1/policy_0_33_valloss-0.98.ptht
# mkdir -p fixed_budget_pjt/checkpoint/3_Ber
# mkdir -p fixed_budget_pjt/checkpoint/3_Normal


