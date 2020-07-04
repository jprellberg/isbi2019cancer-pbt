#!/bin/bash

sbatch gpu.sh run_baseline.py --dataroot /raid/jprellberg/isbi2019cancer/data "$@"
