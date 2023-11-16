# Swin
swin_models=(swin_small_patch4_window7_224.ms_in22k_ft_in1k swin_base_patch4_window7_224.ms_in22k_ft_in1k swin_large_patch4_window7_224.ms_in22k_ft_in1k)
names=(swins swinb swinl)
img_sizes=(224 224 224)

for var in ${!swin_models[@]}
do
echo ${swin_models[$var]}
python swin_quip.py  --exp_name GPTQ \
                    --parent_dir ablation_${names[$var]} \
                    --wbits 2 \
                    --quant ldlq \
                    --pre_gptqH \
                    --qfn a \
                    --eval_batch_size 8 \
                    --timm_model_name ${swin_models[$var]} \
                    --percdamp 0.01  \
                    --img_size ${img_sizes[$var]}
sleep 2
python swin_quip.py  --exp_name GPTQ_TFF \
                    --parent_dir ablation_${names[$var]} \
                    --wbits 2 \
                    --quant ldlq \
                    --pre_gptqH \
                    --pre_tff \
                    --tff_redundancy 1 \
                    --pre_proj \
                    --qfn a \
                    --eval_batch_size 8 \
                    --timm_model_name ${swin_models[$var]} \
                    --percdamp 0.01  \
                    --img_size ${img_sizes[$var]}
sleep 2
python swin_quip.py  --exp_name GPTQ_TFF_CLAMP \
                    --parent_dir ablation_${names[$var]} \
                    --wbits 2 \
                    --quant ldlq \
                    --pre_gptqH \
                    --pre_tff \
                    --tff_redundancy 1 \
                    --pre_proj \
                    --qfn s \
                    --x_sigma 2.0 \
                    --eval_batch_size 8 \
                    --timm_model_name ${swin_models[$var]} \
                    --percdamp 0.01  \
                    --img_size ${img_sizes[$var]}
sleep 2
python swin_quip.py  --exp_name GPTQ_TFF_CLAMP_RED1p1 \
                    --parent_dir ablation_${names[$var]} \
                    --wbits 2 \
                    --quant ldlq \
                    --pre_gptqH \
                    --pre_tff \
                    --tff_redundancy 1.1 \
                    --pre_proj \
                    --qfn s \
                    --x_sigma 2.0 \
                    --eval_batch_size 8 \
                    --timm_model_name ${swin_models[$var]} \
                    --percdamp 0.01  \
                    --img_size ${img_sizes[$var]}
sleep 2
done
