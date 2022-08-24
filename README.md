
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
device=1
pooling=virtual_node
max_steps=1000
max_depth=4
hidden=256
model_cls=rgin
out=/home/v-chaozhang/model/${model_cls}_maxs_${max_steps}_hi_${hidden}_maxd_${max_depth}_po_${pooling}
python main.py \
--trainer-cls LocalTrainer \
--client-cls RGINClient \
--local-optim-cls Adam \
--max-steps ${max_steps} \
--local-epoch 1 \
--dropout 0.2 \
--max-depth ${max_depth} \
--hidden ${hidden} \
--clients-per-step 12 \
--device ${device} \
--out-path ${out} \
--model-cls ${model_cls} \
--pooling ${pooling}
```