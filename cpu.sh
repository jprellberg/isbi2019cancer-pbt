#!/bin/bash

#SBATCH --ntasks=2
#SBATCH --mem=64G
#SBATCH --time=2-00:00
#SBATCH --partition=long

PYTHONPATH="$PYTHONPATH:$(pwd)" python3.6 -u "$@"
