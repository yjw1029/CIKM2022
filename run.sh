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
