max_steps=1000
seed=0
pooling=virtual_node
hiddens=(256 512)
gpu=0
max_depths=(2 4 6 8)

for max_depth in ${max_depths[@]}
do

for hidden in ${hiddens[@]}
do


out_path=/home/v-chaozhang/model/multivirtual_cikm2022-fix_maxs_100_se_0_hi_${hidden}_maxd_${max_depth}_po_${pooling}
echo ${out_path}
echo ${gpu}

python main.py \
--trainer-cls LocalTrainer --client-cls RGNNClient \
--server-cls BaseServer --agg-cls NonUniformAgg \
--global-optim-cls Adam --global-lr 0.001 \
--local-optim-cls Adam --local-epoch 1 \
--max-steps ${max_steps} --local-lr 0.001 \
--pooling ${pooling} --max-depth ${max_depth} --hidden ${hidden} \
--model-cls rgin --dropout 0.2 \
--out-path ${out_path} --local-batch-size 64 \
--device ${gpu} --clients 3 7  --clients-per-step 2 --patient 30 --virtual_node_num 3


gpu=`expr ${gpu} + 1`

done
done