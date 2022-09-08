
max_steps=1000
seed=0
pooling=mean
hiddens=(256 512)
gpu=0
max_depths=(6 8 10)

for max_depth in ${max_depths[@]}
do

for hidden in ${hiddens[@]}
do

load_pretrain_path=/home/v-chaozhang/model/pretrain_rgcn_cikm2022-fix_maxs_100_se_0_hi_${hidden}_maxd_${max_depth}_po_${pooling}
out_path=/home/v-chaozhang/model/load_pretrain_rgcn_cikm2022-fix_maxs_100_se_0_hi_${hidden}_maxd_${max_depth}_po_${pooling}_lr0.008
echo ${out_path}
echo ${gpu}

# python main.py \
# --trainer-cls PretrainTrainer --client-cls PretrainClient \
# --server-cls BaseServer --agg-cls NonUniformAgg \
# --global-optim-cls Adam --global-lr 0.001 \
# --local-optim-cls SGD --local-epoch 1 \
# --max-steps ${max_steps} --local-lr 0.01 \
# --pooling mean --max-depth ${max_depth} --hidden ${hidden} \
# --model-cls rgcn --dropout 0.2 \
# --sample-node-num 1 --sample-depth 3 --sample-neighbor-number 4 \
# --attr-ratio 0.5 --client-config-file /home/v-chaozhang/CIKM2022/config/local_pretrain_per_client.yaml \
# --out-path ${out_path} --local-batch-size 64 --num-bases 10 --reco-steps 20 \
# --eval-steps 1 --param-filter-list "encoder_atom" "encoder" "clf" "comp" "norms" "init_emb" "attr_decoder" "matchers" "link_dec_dict" "neg_queue" --device ${gpu} &

python main.py \
--trainer-cls LocalTrainer --client-cls RGNNClient \
--server-cls BaseServer --agg-cls NonUniformAgg \
--global-optim-cls Adam --global-lr 0.001 \
--local-optim-cls SGD --local-epoch 1 \
--max-steps ${max_steps} --local-lr 0.008 \
--pooling mean --max-depth ${max_depth} --hidden ${hidden} \
--model-cls rgcn --dropout 0.2 \
--out-path ${out_path} --local-batch-size 64 --num-bases 10 --reco-steps 20 \
--eval-steps 1 --param-filter-list "encoder_atom" "encoder" "clf" "comp" "norms" "init_emb" "attr_decoder" "matchers" "link_dec_dict" "neg_queue" --device ${gpu} \
--load_pretrain_path ${load_pretrain_path} --clients 1 2 3 4 5 6 7 8 11 12 --clients-per-step 10 &


gpu=`expr ${gpu} + 1`

done
done