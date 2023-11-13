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

python fullPrecision_exps.py --batch_size 32 --parent_dir full_pres --exp_name vit_huge_patch14_clip_224_laion2b_ft_in1k --model_name vit_huge_patch14_clip_224.laion2b_ft_in1k
sleep 1
python fullPrecision_exps.py --batch_size 32 --parent_dir full_pres --exp_name vit_large_patch16_224_augreg_in21k_ft_in1k --model_name vit_large_patch16_224.augreg_in21k_ft_in1k
sleep 1
