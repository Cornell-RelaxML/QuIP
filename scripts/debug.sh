vit_models=(vit_small_patch16_224)

for var in ${!vit_models[@]}
do
echo ${vit_models[$var]}
python vit_quip.py --exp_name debug_thread --parent_dir quip --wbits 2 --quant ldlq --eval_batch_size 64 --pre_gptqH --pre_proj --pre_proj_extra 0 --pre_rescale --qfn s --x_sigma 2.0 --timm_model_name ${vit_models[$var]} --train_batch_path ./data/train_batch_2.pt
sleep 2
done
