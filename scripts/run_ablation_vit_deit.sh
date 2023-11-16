# ViT
# vit_models=(vit_base_patch16_224)
# img_sizes=(224)
# vit_models=(deit3_small_patch16_224.fb_in22k_ft_in1k deit3_base_patch16_224.fb_in22k_ft_in1k deit3_huge_patch14_224.fb_in22k_ft_in1k)
# names=(deits deitb deith)
# img_sizes=(224 224 224)
vit_models=(vit_huge_patch14_clip_224.laion2b_ft_in1k)
names=(vith)
img_sizes=(224 224 224 224 224 224)
# vit_models=(vit_small_patch16_224 vit_base_patch16_224 vit_huge_patch14_clip_224.laion2b_ft_in1k deit3_small_patch16_224.fb_in22k_ft_in1k deit3_base_patch16_224.fb_in22k_ft_in1k deit3_huge_patch14_224.fb_in22k_ft_in1k)
# names=(vits vitb vith deits deitb deith)
# img_sizes=(224 224 224 224 224 224)

for var in ${!vit_models[@]}
do
echo ${vit_models[$var]}
# python vit_quip.py  --exp_name GPTQ \
#                     --parent_dir ablation_${names[$var]} \
#                     --wbits 2 \
#                     --quant ldlq \
#                     --pre_gptqH \
#                     --qfn a \
#                     --eval_batch_size 8 \
#                     --timm_model_name ${vit_models[$var]} \
#                     --percdamp 0.01  \
#                     --img_size ${img_sizes[$var]}
# sleep 2
# python vit_quip.py  --exp_name GPTQ_TFF \
#                     --parent_dir ablation_${names[$var]} \
#                     --wbits 2 \
#                     --quant ldlq \
#                     --pre_gptqH \
#                     --pre_tff \
#                     --tff_redundancy 1 \
#                     --pre_proj \
#                     --qfn a \
#                     --eval_batch_size 8 \
#                     --timm_model_name ${vit_models[$var]} \
#                     --percdamp 0.01  \
#                     --img_size ${img_sizes[$var]}
# sleep 2
# python vit_quip.py  --exp_name GPTQ_TFF_CLAMP \
#                     --parent_dir ablation_${names[$var]} \
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
# python vit_quip.py  --exp_name GPTQ_TFF_CLAMP_RED1p1 \
#                     --parent_dir ablation_${names[$var]} \
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
python vit_quip.py  --exp_name GPTQ_CLAMP \
                    --parent_dir ablation_${names[$var]} \
                    --wbits 2 \
                    --quant gptq \
                    --pre_gptqH \
                    --qfn s \
                    --x_sigma 2.0 \
                    --eval_batch_size 8 \
                    --timm_model_name ${vit_models[$var]} \
                    --percdamp 0.02  \
                    --img_size ${img_sizes[$var]}
sleep 2
done
