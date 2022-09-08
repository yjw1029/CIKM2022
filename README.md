
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

* RGIN
```bash
python main.py --trainer-cls LocalTrainer --client-cls RGNNClient \
--local-optim-cls Adam --max-steps 1000 --local-epoch 1 \
--dropout 0.2 --max-depth 4 --hidden 256 --clients-per-step 3 \
--model-cls rgin --pooling virtual_node --clients 2 3 5 --patient 30 --mask_node True
```

* Base Decomposition
```bash
python main.py --trainer-cls FedAvgTrainer --client-cls RGNNClient \
--server-cls BaseServer --agg-cls NonUniformAgg \
--global-optim-cls Adam --global-lr 0.001 \
--local-optim-cls Adam --max-steps 100 --local-epoch 1 \
--dropout 0.2 --max-depth 4 --hidden 256 --clients-per-step 13 \
--model-cls rgin --pooling virtual_node --num-bases 5 --base_agg decomposition \
--param-filter-list "encoder_atom" "encoder" "clf" "comp" \
--enable-finetune True --max-ft-steps 100 --ft-lr 0.001 \
--ft-local-optim-cls Adam --ft-local-epoch 1 --out-path /home/v-chaozhang/model/rgin_maxs_1000_hi_256_maxd_4_po_virtual_node_base_5_updatebase_ft --device 1
```

* Base MoE
```bash
python main.py --trainer-cls FedAvgTrainer --client-cls RGNNClient \
--server-cls BaseServer --agg-cls NonUniformAgg \
--global-optim-cls Adam --global-lr 0.001 \
--local-optim-cls Adam --max-steps 100 --local-epoch 1 \
--dropout 0.2 --max-depth 4 --hidden 256 --clients-per-step 13 \
--model-cls rgin --pooling virtual_node --num-bases 5 --base_agg moe \
--param-filter-list "encoder_atom" "encoder" "clf" "comp" \
--enable-finetune True --max-ft-steps 100 --ft-lr 0.001 \
--ft-local-optim-cls Adam --ft-local-epoch 1 
```

* FedAdam + FT
```bash
python main.py --trainer-cls FedAvgTrainer --client-cls BaseClient \
--server-cls BaseServer --agg-cls NonUniformAgg \
--global-optim-cls Adam --global-lr 0.001 \
--local-optim-cls SGD --local-epoch 1 --local-lr 0.01 \
--clients 1 2 3 --clients-per-step 3 --max-steps 100 \
--pooling virtual_node --max-depth 2 --hidden 64 \
--model-cls gin --dropout 0.2 \
--enable-finetune True --max-ft-steps 100 --ft-lr 0.001 \
--ft-local-optim-cls Adam --ft-local-epoch 1
```

* FadAdam + FedBN + FedReco + FT
```bash
python main.py --clients 1 2 3 4 5 6 7 8 11 12 --clients-per-step 10 \
--trainer-cls FedAvgTrainer --client-cls FLRecoClient \
--server-cls BaseServer --agg-cls NonUniformAgg \
--global-optim-cls Adam --global-lr 0.001 \
--local-optim-cls SGD --local-epoch 1 \
--max-steps 100 --local-lr 0.01 \
--pooling virtual_node --max-depth 4 --hidden 256 \
--model-cls gin --dropout 0.2 \
--enable-finetune True --max-ft-steps 100 --ft-lr 0.001 \
--ft-local-optim-cls Adam --ft-local-epoch 1 --reco-steps 20 \
--param-filter-list encoder_atom encoder clf norms
```