#!/bin/bash

#SBATCH --ntasks=6
#SBATCH --mem=30G
#SBATCH --gres=gpu:1
#SBATCH --time=0-02:00
#SBATCH --partition=long

PYTHONPATH="$PYTHONPATH:$(pwd)" python3.6 -u "$@"
