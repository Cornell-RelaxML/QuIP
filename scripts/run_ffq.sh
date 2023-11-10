models=(vit_tiny_patch16_224 vit_small_patch16_224 vit_small_patch32_224 vit_base_patch16_224 deit_tiny_patch16_224 deit_small_patch16_224 deit_base_patch16_224)

for var in ${!models[@]}
do
echo ${models[$var]}
python vit_quip.py --exp_name ${models[$var]} --parent_dir ffq_van --wbits 2 --Weiner_m_diag_rank 0 --quant ldlq --pre_tff --pre_gptqH --qfn s --eval_batch_size 128 --tff_redundancy 2 --x_sigma 2.5 --timm_model_name ${models[$var]}
sleep 2
done