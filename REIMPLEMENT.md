# Reimplementation
We intoduce how to reimplement our solution in details.
However, due to the gpu version of pytorch_geometric is non-deterministic (refer to [issue 92](https://github.com/pyg-team/pytorch_geometric/issues/92) of pytorch_geometric), we are not sure the results are 100% same.
Therefore, we also provide the checkpoints and predictions of all client models and our final submission file for quick and precise reimplementation.


# Getting Started
## Results for clients 1-8 and 11-12

### K-fold train base models
* GINE
```bash
for pooling in virtual_node mean
do
    for hidden in 64 128 256 512
    do
        for max_depth in 2 4 6 8 10
        do
            python main.py --clients 1 2 3 4 5 6 7 8 11 12 \
            --trainer-cls KFoldLocalTrainer --client-cls RGNNClient \
            --local-optim-cls Adam --max-steps 1000 --local-epoch 1 \
            --seed 0 --pooling ${pooling} \
            --max-depth ${max_depth} --hidden ${hidden} --model-cls gine \
            --dropout 0.2 --patient 20 --out-path your/out/path
        done
    done
done
```

* RGCN
```bash
for pooling in virtual_node mean
do
    for hidden in 64 128 256 512
    do
        for max_depth in 2 4 6 8 10
        do
        python main.py --clients 1 2 3 4 5 6 7 8 11 12 \
        --trainer-cls KFoldLocalTrainer --client-cls RGNNClient \
        --local-optim-cls Adam --max-steps 1000 --local-epoch 1 \
        --seed 0 --pooling ${pooling} \
        --max-depth ${max_depth} --hidden ${hidden} --model-cls rgcn \
        --dropout 0.2 --patient 20 --out-path your/out/path
        done
    done
done
```

* RGIN
```bash
for pooling in virtual_node mean
do
    for hidden in 64 128 256 512
    do
        for max_depth in 2 4 6 8 10
        do
        python main.py --clients 1 2 3 4 5 6 7 8 11 12 \
        --trainer-cls KFoldLocalTrainer --client-cls RGNNClient \
        --local-optim-cls Adam --max-steps 1000 --local-epoch 1 \
        --seed 0 --pooling ${pooling} \
        --max-depth ${max_depth} --hidden ${hidden} --model-cls rgin \
        --dropout 0.2 --patient 20 --out-path your/out/path
        done
    done
done
```

* GIN
```bash
for pooling in virtual_node mean
do
    for hidden in 64 128 256 512
    do
        for max_depth in 2 4 6 8 10
        do
        python main.py --clients 1 2 3 4 5 6 7 8 11 12 \
        --trainer-cls KFoldLocalTrainer --client-cls BaseClient \
        --local-optim-cls Adam --max-steps 1000 --local-epoch 1 \
        --seed 0 --pooling ${pooling} \
        --max-depth ${max_depth} --hidden ${hidden} --model-cls gin \
        --dropout 0.2 --patient 20 --out-path your/out/path
        done
    done
done
```

### Ensemble results
```bash
python merge_result.py --mode kfold --out-path your/out/path --soft True --topk_min 20 --topk_max 40 --clients 1 2 3 4 5 6 7 8 11 12 --save-path your/save/path
```


## Results for clients 9 10 13
Since the sample size of clients 9 10 and 13 is too large, we only run fold 0 and 1 results for them.
We believe the results will get higher if experiments for fold 2 and 3 completes.

### K-fold train base models
* GINE
```bash
for pooling in virtual_node mean
do
    for hidden in 64 128 256 512
    do
        for max_depth in 2 4 6 8 10
        do
            for val_fold in 0 1
            do
                python main.py --clients 1 2 3 4 5 6 7 8 9 10 11 12 13 \
                --trainer-cls LocalTrainer --client-cls RGNNClient \
                --local-optim-cls Adam --max-steps 1000 --local-epoch 1 \
                --seed 0 --pooling ${pooling} --val-fold ${val_fold} \
                --max-depth ${max_depth} --hidden ${hidden} --model-cls gine \
                --dropout 0.2 --patient 20 --out-path your/out/path
            done
        done
    done
done
```

* RGCN
```bash
for pooling in virtual_node mean
do
    for hidden in 64 128 256 512
    do
        for max_depth in 2 4 6 8 10
        do
            for val_fold in 0 1
            do
            python main.py --clients 1 2 3 4 5 6 7 8 9 10 11 12 13 \
            --trainer-cls KFoldLocalTrainer --client-cls RGNNClient \
            --local-optim-cls Adam --max-steps 1000 --local-epoch 1 \
            --seed 0 --pooling ${pooling} --val-fold ${val_fold} \
            --max-depth ${max_depth} --hidden ${hidden} --model-cls rgcn \
            --dropout 0.2 --patient 20 --out-path your/out/path
            done
        done
    done
done
```

* RGIN
```bash
for pooling in virtual_node mean
do
    for hidden in 64 128 256 512
    do
        for max_depth in 2 4 6 8 10
        do
            for val_fold in 0 1
            do
            python main.py --clients 1 2 3 4 5 6 7 8 9 10 11 12 13 \
            --trainer-cls KFoldLocalTrainer --client-cls RGNNClient \
            --local-optim-cls Adam --max-steps 1000 --local-epoch 1 \
            --seed 0 --pooling ${pooling} --val-fold ${val_fold} \
            --max-depth ${max_depth} --hidden ${hidden} --model-cls rgin \
            --dropout 0.2 --patient 20 --out-path your/out/path
            done
        done
    done
done
```

* GIN
```bash
for pooling in virtual_node mean
do
    for hidden in 64 128 256 512
    do
        for max_depth in 2 4 6 8 10
        do
            for val_fold in 0 1
            do
            python main.py --clients 1 2 3 4 5 6 7 8 9 10 11 12 13 \
            --trainer-cls KFoldLocalTrainer --client-cls BaseClient \
            --local-optim-cls Adam --max-steps 1000 --local-epoch 1 \
            --seed 0 --pooling ${pooling} --val-fold ${val_fold} \
            --max-depth ${max_depth} --hidden ${hidden} --model-cls gin \
            --dropout 0.2 --patient 20 --out-path your/out/path
            done
        done
    done
done
```

### Ensemble results
```bash
python merge_result.py --mode kfold --out-path your/out/path --soft True --topk_min 0 --topk_max 10 --clients 9 10 13 --save-path your/save/path
```

## Optimization for clients 2


# Quick Check