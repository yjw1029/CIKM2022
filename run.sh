device=0
out=/home/v-chaozhang/model/RGIN


python main.py \
--trainer-cls LocalTrainer \
--client-cls RGINClient \
--local-optim-cls Adam \
--max-steps 1000 \
--local-epoch 1 \
--dropout 0.2 \
--max-depth 4 \
--hidden 256 \
--clients-per-step 12 \
--device ${device} \
--out-path ${out} \
--model-cls rgin \
--pooling mean
