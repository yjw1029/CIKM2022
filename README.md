
# Environment
registry: docker.io

image: yjw1029/singularity:fs-val-202208150247

# Command to run
```bash
# Isolated Training
python main.py --trainer-cls LocalTrainer --client-cls BaseClient \
      --local-optim-cls Adam --max-steps 100 --local-epoch 1 --clients-num 13
```