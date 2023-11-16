# python fullPrecision_exps.py --batch_size 128 --parent_dir full_pres --exp_name vit_tiny_patch16_224 --model_name vit_tiny_patch16_224
# sleep 1
# python fullPrecision_exps.py --batch_size 128 --parent_dir full_pres --exp_name vit_small_patch32_224 --model_name vit_small_patch32_224
# sleep 1
# python fullPrecision_exps.py --batch_size 128 --parent_dir full_pres --exp_name vit_small_patch16_224 --model_name vit_small_patch16_224
# sleep 1
# python fullPrecision_exps.py --batch_size 128 --parent_dir full_pres --exp_name vit_base_patch16_224 --model_name vit_base_patch16_224
# sleep 1
# python fullPrecision_exps.py --batch_size 128 --parent_dir full_pres --exp_name deit_tiny_patch16_224 --model_name deit_tiny_patch16_224
# sleep 1
# python fullPrecision_exps.py --batch_size 128 --parent_dir full_pres --exp_name deit_small_patch16_224 --model_name deit_small_patch16_224
# sleep 1
# python fullPrecision_exps.py --batch_size 128 --parent_dir full_pres --exp_name deit_base_patch16_224 --model_name deit_base_patch16_224
# sleep 1
# python fullPrecision_exps.py --batch_size 128 --parent_dir full_pres --exp_name swin_small_patch4_window7_224 --model_name swin_small_patch4_window7_224
# sleep 1
# python fullPrecision_exps.py --batch_size 128 --parent_dir full_pres --exp_name swin_base_patch4_window7_224 --model_name swin_base_patch4_window7_224
# sleep 1
# python fullPrecision_exps.py --batch_size 128 --parent_dir full_pres --exp_name swin_base_patch4_window12_384 --model_name swin_base_patch4_window12_384
# sleep 1

# python fullPrecision_exps.py --batch_size 32 --parent_dir full_pres --exp_name vit_huge_patch14_clip_224_laion2b_ft_in1k --model_name vit_huge_patch14_clip_224.laion2b_ft_in1k
# sleep 1
# python fullPrecision_exps.py --batch_size 32 --parent_dir full_pres --exp_name vit_large_patch16_224_augreg_in21k_ft_in1k --model_name vit_large_patch16_224.augreg_in21k_ft_in1k
# sleep 1

# python fullPrecision_exps.py --batch_size 32 --parent_dir full_pres --exp_name beit_base_patch16_384.in22k_ft_in22k_in1k --model_name beit_base_patch16_384.in22k_ft_in22k_in1k
# sleep 1
# python fullPrecision_exps.py --batch_size 32 --parent_dir full_pres --exp_name beit_large_patch16_512.in22k_ft_in22k_in1k --model_name beit_large_patch16_512.in22k_ft_in22k_in1k
# sleep 1

# python fullPrecision_exps.py --batch_size 32 --parent_dir full_pres --exp_name beit_base_patch16_224.in22k_ft_in22k_in1k --model_name beit_base_patch16_384.in22k_ft_in22k_in1k
# sleep 1
# python fullPrecision_exps.py --batch_size 32 --parent_dir full_pres --exp_name beit_large_patch16_224.in22k_ft_in22k_in1k --model_name beit_large_patch16_512.in22k_ft_in22k_in1k
# sleep 1

# # ViT
# vit_models=(vit_tiny_patch16_224 vit_small_patch16_224 vit_medium_patch16_gap_256.sw_in12k_ft_in1k vit_base_patch16_224 vit_large_patch16_224.augreg_in21k_ft_in1k vit_huge_patch14_clip_224.laion2b_ft_in1k)
# img_sizes=(224 224 256 224 224 224)
# for var in ${!vit_models[@]}
# do
# echo ${vit_models[$var]}
# python fullPrecision_exps.py --batch_size 8 --parent_dir full_pres --exp_name ${vit_models[$var]} --model_name ${vit_models[$var]}  --img_size ${img_sizes[$var]}
# sleep 2
# done

# # DeiT
# deit_models=(deit3_small_patch16_224.fb_in22k_ft_in1k deit3_medium_patch16_224.fb_in22k_ft_in1k deit3_base_patch16_224.fb_in22k_ft_in1k deit3_large_patch16_224.fb_in22k_ft_in1k deit3_huge_patch14_224.fb_in22k_ft_in1k)
# img_sizes=(224 224 224 224 224)
# 
# for var in ${!deit_models[@]}
# do
# echo ${deit_models[$var]}
# python fullPrecision_exps.py --batch_size 8 --parent_dir full_pres --exp_name ${deit_models[$var]} --model_name ${deit_models[$var]} --img_size ${img_sizes[$var]}
# sleep 2
# done

# # Swin
swin_models=(swin_tiny_patch4_window7_224.ms_in22k_ft_in1k swin_small_patch4_window7_224.ms_in22k_ft_in1k swin_large_patch4_window7_224.ms_in22k_ft_in1k swin_base_patch4_window7_224.ms_in22k_ft_in1k)
img_sizes=(224 224 224 224)

for var in ${!swin_models[@]}
do
echo ${swin_models[$var]}
python fullPrecision_exps.py --batch_size 8 --parent_dir full_pres --exp_name ${swin_models[$var]} --model_name ${swin_models[$var]} --img_size ${img_sizes[$var]}
sleep 2
done

cait_models=(cait_xxs24_224.fb_dist_in1k cait_xxs36_224.fb_dist_in1k cait_s24_224.fb_dist_in1k)
img_sizes=(224 224 224)

for var in ${!cait_models[@]}
do
echo ${cait_models[$var]}
python fullPrecision_exps.py --batch_size 8 --parent_dir full_pres --exp_name ${cait_models[$var]} --model_name ${cait_models[$var]} --img_size ${img_sizes[$var]}
sleep 2
done

cait_models=(cait_xs24_384.fb_dist_in1k cait_s36_384.fb_dist_in1k cait_m36_384.fb_dist_in1k)
img_sizes=(384 384 384)

for var in ${!cait_models[@]}
do
echo ${cait_models[$var]}
python fullPrecision_exps.py --batch_size 4 --parent_dir full_pres --exp_name ${cait_models[$var]} --model_name ${cait_models[$var]} --img_size ${img_sizes[$var]}
sleep 2
done

cait_models=(cait_m48_448.fb_dist_in1k)
img_sizes=(448)

for var in ${!cait_models[@]}
do
echo ${cait_models[$var]}
python fullPrecision_exps.py --batch_size 1 --parent_dir full_pres --exp_name ${cait_models[$var]} --model_name ${cait_models[$var]} --img_size ${img_sizes[$var]}
sleep 2
done