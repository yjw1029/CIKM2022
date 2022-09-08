max_steps=1000
seed=0
pooling=mean
hiddens=(256)
gpu=2
max_depths=(1 2 4 6 8 10)

for max_depth in ${max_depths[@]}
do

for hidden in ${hiddens[@]}
do


out_path=/home/v-chaozhang/model/film_cikm2022-fix_maxs_100_se_0_hi_${hidden}_maxd_${max_depth}_po_${pooling}
echo ${out_path}
echo ${gpu}

python main.py \
--trainer-cls LocalTrainer --client-cls RGNNClient \
--server-cls BaseServer --agg-cls NonUniformAgg \
--global-optim-cls Adam --global-lr 0.001 \
--local-optim-cls Adam --local-epoch 1 \
--max-steps ${max_steps} --local-lr 0.01 \
--pooling ${pooling} --max-depth ${max_depth} --hidden ${hidden} \
--model-cls film --dropout 0.2 \
--out-path ${out_path} --local-batch-size 64 \
--device ${gpu} --clients 2 3 5 6 7 11 12 --clients-per-step 7 --patient 30 &


gpu=`expr ${gpu} + 1`

done
done