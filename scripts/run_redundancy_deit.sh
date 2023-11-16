# DeiT 
vit_models=(deit3_small_patch16_224.fb_in22k_ft_in1k deit3_base_patch16_224.fb_in22k_ft_in1k deit3_huge_patch14_224.fb_in22k_ft_in1k)
names=(deits deitb deith)
img_sizes=(224 224 224)
# vit_models=(deit3_small_patch16_224.fb_in22k_ft_in1k deit3_base_patch16_224.fb_in22k_ft_in1k)
# names=(deits deitb)
# img_sizes=(224 224)

for var in ${!vit_models[@]}
do
echo ${vit_models[$var]}
# python vit_quip.py  --exp_name red_1 \
#                     --parent_dir redundancy_${names[$var]} \
#                     --wbits 2 \
#                     --quant ldlq \
#                     --pre_gptqH \
#                     --pre_tff \
#                     --tff_redundancy 1 \
#                     --pre_proj \
#                     --qfn s \
#                     --x_sigma 2.0 \
#                     --eval_batch_size 8 \
#                     --timm_model_name ${vit_models[$var]} \
#                     --percdamp 0.01  \
#                     --img_size ${img_sizes[$var]}
# sleep 2
# python vit_quip.py  --exp_name red_1p05 \
#                     --parent_dir redundancy_${names[$var]} \
#                     --wbits 2 \
#                     --quant ldlq \
#                     --pre_gptqH \
#                     --pre_tff \
#                     --tff_redundancy 1.05 \
#                     --pre_proj \
#                     --qfn s \
#                     --x_sigma 2.0 \
#                     --eval_batch_size 8 \
#                     --timm_model_name ${vit_models[$var]} \
#                     --percdamp 0.01  \
#                     --img_size ${img_sizes[$var]}
# sleep 2
# python vit_quip.py  --exp_name red_1p1 \
#                     --parent_dir redundancy_${names[$var]} \
#                     --wbits 2 \
#                     --quant ldlq \
#                     --pre_gptqH \
#                     --pre_tff \
#                     --tff_redundancy 1.1 \
#                     --pre_proj \
#                     --qfn s \
#                     --x_sigma 2.0 \
#                     --eval_batch_size 8 \
#                     --timm_model_name ${vit_models[$var]} \
#                     --percdamp 0.01  \
#                     --img_size ${img_sizes[$var]}
# sleep 2
# python vit_quip.py  --exp_name red_1p15 \
#                     --parent_dir redundancy_${names[$var]} \
#                     --wbits 2 \
#                     --quant ldlq \
#                     --pre_gptqH \
#                     --pre_tff \
#                     --tff_redundancy 1.15 \
#                     --pre_proj \
#                     --qfn s \
#                     --x_sigma 2.0 \
#                     --eval_batch_size 8 \
#                     --timm_model_name ${vit_models[$var]} \
#                     --percdamp 0.01  \
#                     --img_size ${img_sizes[$var]}
# sleep 2
# python vit_quip.py  --exp_name red_1p2 \
#                     --parent_dir redundancy_${names[$var]} \
#                     --wbits 2 \
#                     --quant ldlq \
#                     --pre_gptqH \
#                     --pre_tff \
#                     --tff_redundancy 1.2 \
#                     --pre_proj \
#                     --qfn s \
#                     --x_sigma 2.0 \
#                     --eval_batch_size 8 \
#                     --timm_model_name ${vit_models[$var]} \
#                     --percdamp 0.01  \
#                     --img_size ${img_sizes[$var]}
# sleep 2
# python vit_quip.py  --exp_name red_1p25 \
#                     --parent_dir redundancy_${names[$var]} \
#                     --wbits 2 \
#                     --quant ldlq \
#                     --pre_gptqH \
#                     --pre_tff \
#                     --tff_redundancy 1.25 \
#                     --pre_proj \
#                     --qfn s \
#                     --x_sigma 2.0 \
#                     --eval_batch_size 8 \
#                     --timm_model_name ${vit_models[$var]} \
#                     --percdamp 0.01  \
#                     --img_size ${img_sizes[$var]}
# sleep 2
python vit_quip.py  --exp_name red_1p3 \
                    --parent_dir redundancy_${names[$var]} \
                    --wbits 2 \
                    --quant ldlq \
                    --pre_gptqH \
                    --pre_tff \
                    --tff_redundancy 1.3 \
                    --pre_proj \
                    --qfn s \
                    --x_sigma 2.0 \
                    --eval_batch_size 8 \
                    --timm_model_name ${vit_models[$var]} \
                    --percdamp 0.01  \
                    --img_size ${img_sizes[$var]}
sleep 2
done
