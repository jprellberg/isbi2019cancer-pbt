#!/bin/bash

sbatch gpu.sh run_test.py --dataroot /raid/jprellberg/isbi2019cancer/data "$@"
