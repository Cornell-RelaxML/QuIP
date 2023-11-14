# vit_models=(vit_tiny_patch16_224 vit_small_patch16_224 vit_small_patch32_224 vit_base_patch16_224 deit_tiny_patch16_224 deit_small_patch16_224 deit_base_patch16_224)
# 
# for var in ${!vit_models[@]}
# do
# echo ${vit_models[$var]}
# python vit_quip.py --exp_name ${vit_models[$var]} --parent_dir ffq_red_1p1 --wbits 2 --Weiner_m_diag_rank 0 --quant ldlq --pre_tff --pre_gptqH --qfn s --eval_batch_size 64  --tff_redundancy 1.1 --x_sigma 2.0 --timm_model_name ${vit_models[$var]} --train_batch_path ./data/train_batch_2.pt --pre_proj --percdamp 0.02
# sleep 2
# done
# 
# swin_models=(swin_small_patch4_window7_224 swin_base_patch4_window7_224 swin_base_patch4_window12_384)
# 
# for var in ${!swin_models[@]}
# do
# echo ${swin_models[$var]}
# python swin_quip.py --exp_name ${swin_models[$var]} --parent_dir ffq_red_1p1 --wbits 2 --Weiner_m_diag_rank 0 --quant ldlq --pre_tff --pre_gptqH --qfn s --eval_batch_size 16 --tff_redundancy 1.1 --x_sigma 2.0 --timm_model_name ${swin_models[$var]} --train_batch_path ./data/train_batch_2.pt --pre_proj --percdamp 0.01
# sleep 2
# done

vit_models=(vit_huge_patch14_clip_224.laion2b_ft_in1k vit_large_patch16_224.augreg_in21k_ft_in1k)

for var in ${!vit_models[@]}
do
echo ${vit_models[$var]}
python vit_quip.py --exp_name ${vit_models[$var]} --parent_dir ffq_red_1p1 --wbits 2 --Weiner_m_diag_rank 0 --quant ldlq --pre_tff --pre_gptqH --qfn s --eval_batch_size 8 --tff_redundancy 1.1 --x_sigma 2.0 --timm_model_name ${vit_models[$var]} --train_batch_path ./data/train_batch_2.pt --pre_proj --percdamp 0.02
sleep 2
done
