max_steps=1000
seed=0
pooling=virtual_node
hiddens=(64 128 256 512)
gpu=0
max_depths=(2)

for max_depth in ${max_depths[@]}
do

for hidden in ${hiddens[@]}
do

out_path=/home/v-chaozhang/model/search_rgin_cikm2022-fix_maxs_1000_se_0_hi_${hidden}_maxd_${max_depth}_po_${pooling}
echo ${out_path}
echo ${gpu}

python main.py --trainer-cls LocalTrainer --client-cls RGNNClient \
--local-optim-cls Adam --max-steps ${max_steps} --local-epoch 1 \
--clients-per-step 13 --seed ${seed} --pooling ${pooling} \
--max-depth ${max_depth} --hidden ${hidden} --dropout 0.2 \
--model-cls rgin --patient 20 --device ${gpu} --out-path ${out_path} &

gpu=`expr ${gpu} + 1`

done
done