# # ViT
# vit_models=(vit_tiny_patch16_224 vit_small_patch16_224 vit_medium_patch16_gap_256.sw_in12k_ft_in1k vit_base_patch16_224 vit_large_patch16_224.augreg_in21k_ft_in1k vit_huge_patch14_clip_224.laion2b_ft_in1k)
# img_sizes=(224 224 224 224 224 224)
# 
# for var in ${!vit_models[@]}
# do
# echo ${vit_models[$var]}
# python vit_quip.py --exp_name ${vit_models[$var]} --parent_dir ffq_256  --train_batch_size 256 --wbits 2 --Weiner_m_diag_rank 0 --quant ldlq --pre_tff --pre_gptqH --qfn s --eval_batch_size 8 --tff_redundancy 1 --x_sigma 2.0 --timm_model_name ${vit_models[$var]} --pre_proj --percdamp 0.01  --img_size ${img_sizes[$var]}
# sleep 2
# done

# Swin
swin_models=(swin_small_patch4_window7_224.ms_in22k_ft_in1k swin_large_patch4_window7_224.ms_in22k_ft_in1k swin_base_patch4_window7_224.ms_in22k_ft_in1k)
img_sizes=(224 224 224)

for var in ${!swin_models[@]}
do
echo ${swin_models[$var]}
python swin_quip.py --exp_name ${swin_models[$var]} --parent_dir ffq_3p0sig --wbits 2 --Weiner_m_diag_rank 0 --quant ldlq --pre_tff --pre_gptqH --qfn s --eval_batch_size 8 --tff_redundancy 1 --x_sigma 3.0 --timm_model_name ${swin_models[$var]} --pre_proj --percdamp 0.01  --img_size ${img_sizes[$var]}
sleep 2
done

# # cait_models=(cait_xxs24_224.fb_dist_in1k cait_xxs36_224.fb_dist_in1k cait_xs24_384.fb_dist_in1k cait_s36_384.fb_dist_in1k cait_s24_224.fb_dist_in1k cait_m36_384.fb_dist_in1k cait_m48_448.fb_dist_in1k)
# # img_sizes=(224 224 384 384 224 384 448)
# cait_models=(cait_m48_448.fb_dist_in1k)
# img_sizes=(448)
# 
# for var in ${!cait_models[@]}
# do
# echo ${cait_models[$var]}
# python cait_quip.py --exp_name ${cait_models[$var]} --parent_dir ffq_van --wbits 2 --Weiner_m_diag_rank 0 --quant ldlq --pre_tff --pre_gptqH --qfn s --eval_batch_size 1 --tff_redundancy 1 --x_sigma 2.0 --timm_model_name ${cait_models[$var]} --pre_proj --percdamp 0.01  --img_size ${img_sizes[$var]}
# sleep 2
# done

# DeiT
# deit_models=(deit3_small_patch16_224.fb_in22k_ft_in1k deit3_medium_patch16_224.fb_in22k_ft_in1k deit3_base_patch16_224.fb_in22k_ft_in1k deit3_large_patch16_224.fb_in22k_ft_in1k deit3_huge_patch14_224.fb_in22k_ft_in1k)
# img_sizes=(224 224 224 224 224)
# 
# for var in ${!deit_models[@]}
# do
# echo ${deit_models[$var]}
# python vit_quip.py --exp_name ${deit_models[$var]} --parent_dir ffq_van --wbits 2 --Weiner_m_diag_rank 0 --quant ldlq --pre_tff --pre_gptqH --qfn s --eval_batch_size 8 --tff_redundancy 1 --x_sigma 2.0 --timm_model_name ${deit_models[$var]} --pre_proj --percdamp 0.01  --img_size ${img_sizes[$var]}
# sleep 2
# done

# ViT
# vit_models=(vit_tiny_patch16_224 vit_small_patch16_224 vit_medium_patch16_gap_256.sw_in12k_ft_in1k vit_base_patch16_224 vit_large_patch16_224.augreg_in21k_ft_in1k vit_huge_patch14_clip_224.laion2b_ft_in1k)
# img_sizes=(224 224 224 224 224 224)
# 
# for var in ${!vit_models[@]}
# do
# echo ${vit_models[$var]}
# python vit_quip.py --exp_name ${vit_models[$var]} --parent_dir ffq_van --wbits 2 --Weiner_m_diag_rank 0 --quant ldlq --pre_tff --pre_gptqH --qfn s --eval_batch_size 8 --tff_redundancy 1 --x_sigma 2.0 --timm_model_name ${vit_models[$var]} --pre_proj --percdamp 0.01  --img_size ${img_sizes[$var]}
# sleep 2
# done

# vit_models=(vit_tiny_patch16_224 vit_small_patch16_224 vit_small_patch32_224 vit_base_patch16_224 deit_tiny_patch16_224 deit_small_patch16_224 deit_base_patch16_224)
# 
# for var in ${!vit_models[@]}
# do
# echo ${vit_models[$var]}
# python vit_quip.py --exp_name ${vit_models[$var]} --parent_dir "ffq_van" --wbits 2 --Weiner_m_diag_rank 0 --quant ldlq --pre_tff --pre_gptqH --qfn s --eval_batch_size 64  --tff_redundancy 1 --x_sigma 2.0 --timm_model_name ${vit_models[$var]} --train_batch_path ./data/train_batch_2.pt --pre_proj
# sleep 2
# done

# swin_models=(swin_small_patch4_window7_224 swin_base_patch4_window7_224 swin_base_patch4_window12_384)
# 
# for var in ${!swin_models[@]}
# do
# echo ${swin_models[$var]}
# python swin_quip.py --exp_name ${swin_models[$var]} --parent_dir "ffq_van" --wbits 2 --Weiner_m_diag_rank 0 --quant ldlq --pre_tff --pre_gptqH --qfn s --eval_batch_size 16 --tff_redundancy 1 --x_sigma 2.0 --timm_model_name ${swin_models[$var]} --train_batch_path ./data/train_batch_2.pt --pre_proj
# sleep 2
# done

# # vit_models=(vit_huge_patch14_clip_224.laion2b_ft_in1k vit_large_patch16_224.augreg_in21k_ft_in1k)
# vit_models=(beit_base_patch16_224.in22k_ft_in22k_in1k beit_large_patch16_224.in22k_ft_in22k_in1k)
# 
# for var in ${!vit_models[@]}
# do
# echo ${vit_models[$var]}
# python vit_quip.py --exp_name ${vit_models[$var]} --parent_dir "ffq_van" --wbits 2 --Weiner_m_diag_rank 0 --quant ldlq --pre_tff --pre_gptqH --qfn s --eval_batch_size 8  --tff_redundancy 1 --x_sigma 2.0 --timm_model_name ${vit_models[$var]} --train_batch_path ./data/train_batch_2.pt --pre_proj
# sleep 2
# done