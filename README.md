
# Environment
registry: docker.io

image: yjw1029/singularity:fs-val-202208150247

# Command to run
* Isolated Training
```bash
python main.py --trainer-cls LocalTrainer --client-cls BaseClient \
      --local-optim-cls Adam --max-steps 100 --local-epoch 1 --clients-num 13
```

* FedAvg
```bash
python main.py --trainer-cls FedAvgTrainer --client-cls BaseClient \
      --server-cls BaseServer --agg-cls NonUniformAgg \
      --global-optim-cls Adam --global-lr 0.01 \
      --local-optim-cls SGD --local-epoch 1 --local-lr 0.01 \
      --clients-num 13 --clients-per-step 13 --max-steps 100
```

* Best isolated training (51.1)
```bash
python main.py --trainer-cls LocalTrainer \
    --local-optim-cls Adam --max-steps 1000 --local-epoch 1 \
    --clients-per-step 13 --seed 100 --dropout 0.2 \
    --client-config-file ./config/local_best_per_client.yaml
```

# RGIN
```bash
python main.py --trainer-cls LocalTrainer --client-cls RGNNClient \
--local-optim-cls Adam --max-steps 1000 --local-epoch 1 \
--dropout 0.2 --max-depth 4 --hidden 256 --clients-per-step 13 \
--model-cls rgin --pooling virtual_node
```

# Base Decomposition
```bash
python main.py --trainer-cls FedAvgTrainer --client-cls RGNNClient \
--server-cls BaseServer --agg-cls NonUniformAgg \
--global-optim-cls Adam --global-lr 0.001 \
--local-optim-cls Adam --max-steps 100 --local-epoch 1 \
--dropout 0.2 --max-depth 4 --hidden 256 --clients-per-step 13 \
--model-cls rgin --pooling virtual_node --num-bases 5 \
--param-filter-list "encoder_atom" "encoder" "clf" "comp" 
```