sbatch cpu.sh run_pbt.py server --addr "tcp://*:5577"
sleep 15
for i in {1..8}; do
    sbatch gpu.sh run_pbt.py client \
        --dataroot /raid/jprellberg/isbi2019cancer/data \
        --addr "tcp://localhost:5577"
done
