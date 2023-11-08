# python vit.py --coef_est_type weiner --exp_name van_gptq_16 --wbits 16
# python vit.py --coef_est_type weiner --exp_name van_gptq_4 --wbits 4
# python vit.py --coef_est_type weiner --exp_name van_gptq_3 --wbits 3
# python vit.py --coef_est_type weiner --exp_name van_gptq_2 --wbits 2

# python vit.py --coef_est_type weiner --exp_name tff_gptq_16 --wbits 16
# python vit.py --coef_est_type weiner --exp_name tff_gptq_4 --wbits 4
# python vit.py --coef_est_type weiner --exp_name tff_gptq_3 --wbits 3
# python vit.py --coef_est_type weiner --exp_name tff_gptq_2 --wbits 2

# python vit.py --coef_est_type weiner --exp_name full_weiner_tff_gptq_16 --wbits 16
# python vit.py --coef_est_type weiner --exp_name full_weiner_tff_gptq_4 --wbits 4
# python vit.py --coef_est_type weiner --exp_name full_weiner_tff_gptq_3 --wbits 3
# python vit.py --coef_est_type weiner --exp_name full_weiner_tff_gptq_2 --wbits 2

# python vit_quip.py --coef_est_type weiner --exp_name debug_thread --wbits 2 --Weiner_m_diag_rank 0 --quant ldlq --incoh_processing
# python vit_quip.py --coef_est_type weiner --exp_name debug_thread --wbits 2 --Weiner_m_diag_rank 0 --quant gptq --incoh_processing
# python vit_quip.py --coef_est_type weiner --exp_name debug_thread --wbits 2 --Weiner_m_diag_rank 0 --quant ldlqRG --incoh_processing
# python vit_quip.py --coef_est_type weiner --exp_name debug_thread --wbits 2 --Weiner_m_diag_rank 0 --quant nearest --incoh_processing
# python vit_quip.py --coef_est_type weiner --exp_name debug_thread --wbits 2 --Weiner_m_diag_rank 0 --quant ldlq --qfn a --pre_gptqH
# python vit_quip.py --coef_est_type weiner --exp_name debug_thread --wbits 2 --Weiner_m_diag_rank 0 --quant ldlq --qfn b --pre_proj  
# python vit_quip.py --coef_est_type weiner --exp_name debug_thread --wbits 2 --Weiner_m_diag_rank 0 --quant ldlq --qfn a --pre_rescale
# python vit_quip.py --coef_est_type weiner --exp_name debug_thread --wbits 2 --Weiner_m_diag_rank 0 --quant ldlq --qfn a --pre_rescale --pre_gptqH
# python vit_quip.py --coef_est_type weiner --exp_name debug_thread --wbits 2 --Weiner_m_diag_rank 0 --quant ldlq --qfn b --pre_rescale --pre_proj
# python vit_quip.py --coef_est_type weiner --exp_name debug_thread --wbits 2 --Weiner_m_diag_rank 0 --quant ldlq --qfn a --pre_proj --pre_gptqH
# python vit_quip.py --coef_est_type weiner --exp_name debug_thread --wbits 2 --Weiner_m_diag_rank 0 --quant gptq --qfn b --pre_proj

# TFF based pre processing
# python vit_quip.py --coef_est_type weiner --exp_name debug_thread --wbits 2 --Weiner_m_diag_rank 0 --quant ldlq --qfn a --pre_gptqH
# python vit_quip.py --coef_est_type weiner --exp_name debug_thread --wbits 2 --Weiner_m_diag_rank 0 --quant ldlq --qfn a --pre_gptqH --pre_rescale
# python vit_quip.py --coef_est_type weiner --exp_name debug_thread --wbits 2 --Weiner_m_diag_rank 0 --quant ldlq --qfn a --pre_gptqH --pre_rescale --pre_proj
# python vit_quip.py --coef_est_type weiner --exp_name debug_thread --wbits 2 --Weiner_m_diag_rank 0 --quant ldlq --qfn b --pre_gptqH --pre_proj

# python vit_quip.py --coef_est_type weiner --exp_name debug_thread --wbits 2 --Weiner_m_diag_rank 0 --quant ldlq --qfn b --pre_gptqH

# # different TFF versions
python vit_quip.py --coef_est_type weiner --exp_name debug_thread --wbits 2 --Weiner_m_diag_rank 0 --quant ldlq --pre_tff --pre_gptqH --qfn a
# python vit_quip.py --coef_est_type weiner --exp_name debug_thread --wbits 2 --Weiner_m_diag_rank 0 --quant ldlq --pre_tff --pre_gptqH --qfn a
# python vit_quip.py --coef_est_type weiner --exp_name debug_thread --wbits 2 --Weiner_m_diag_rank 0 --quant ldlq --pre_proj --pre_gptqH --qfn b
# # full quip
# python vit_quip.py --exp_name debug_thread --wbits 2 --quant ldlq --incoh_processing
# # GPTQ
# python vit_quip.py --coef_est_type weiner --exp_name debug_thread --wbits 2 --Weiner_m_diag_rank 0 --quant gptq --qfn a --pre_gptqH

# python vit.py --coef_est_type weiner --exp_name fullW_tff_vitFT_16 --wbits 16 --dataset inet1k_birds --dataset_dir data/inet1k_classes/birds \
#  --ckpt_path output/olvi1_outputs/inet1k_birds-2023-10-02-22-25-22/inet1k_birds_final_ckpt.bin --parent_dir birds_vitFT 
# python vit.py --coef_est_type weiner --exp_name fullW_tff_vitFT_4 --wbits 4 --dataset inet1k_birds --dataset_dir data/inet1k_classes/birds \
#  --ckpt_path output/olvi1_outputs/inet1k_birds-2023-10-02-22-25-22/inet1k_birds_final_ckpt.bin --parent_dir birds_vitFT 
# python vit.py --coef_est_type weiner --exp_name fullW_tff_vitFT_3 --wbits 3 --dataset inet1k_birds --dataset_dir data/inet1k_classes/birds \
#  --ckpt_path output/olvi1_outputs/inet1k_birds-2023-10-02-22-25-22/inet1k_birds_final_ckpt.bin --parent_dir birds_vitFT 
# python vit.py --coef_est_type weiner --exp_name fullW_tff_vitFT_2 --wbits 2 --dataset inet1k_birds --dataset_dir data/inet1k_classes/birds \
#  --ckpt_path output/olvi1_outputs/inet1k_birds-2023-10-02-22-25-22/inet1k_birds_final_ckpt.bin --parent_dir birds_vitFT 

# python vit.py --coef_est_type weiner --exp_name gptq_vitFT_16 --wbits 16 --dataset inet1k_birds --dataset_dir data/inet1k_classes/birds \
#  --ckpt_path output/olvi1_outputs/inet1k_birds-2023-10-02-22-25-22/inet1k_birds_final_ckpt.bin --parent_dir birds_vitFT 
# python vit.py --coef_est_type weiner --exp_name gptq_vitFT_4 --wbits 4 --dataset inet1k_birds --dataset_dir data/inet1k_classes/birds \
#  --ckpt_path output/olvi1_outputs/inet1k_birds-2023-10-02-22-25-22/inet1k_birds_final_ckpt.bin --parent_dir birds_vitFT 
# python vit.py --coef_est_type weiner --exp_name gptq_vitFT_3 --wbits 3 --dataset inet1k_birds --dataset_dir data/inet1k_classes/birds \
#  --ckpt_path output/olvi1_outputs/inet1k_birds-2023-10-02-22-25-22/inet1k_birds_final_ckpt.bin --parent_dir birds_vitFT 
# python vit.py --coef_est_type weiner --exp_name gptq_vitFT_2 --wbits 2 --dataset inet1k_birds --dataset_dir data/inet1k_classes/birds \
#  --ckpt_path output/olvi1_outputs/inet1k_birds-2023-10-02-22-25-22/inet1k_birds_final_ckpt.bin --parent_dir birds_vitFT 

# python vit.py --coef_est_type weiner --exp_name tff_vitFT_16 --wbits 16 --dataset inet1k_birds --dataset_dir data/inet1k_classes/birds \
#  --ckpt_path output/olvi1_outputs/inet1k_birds-2023-10-02-22-25-22/inet1k_birds_final_ckpt.bin --parent_dir birds_vitFT 
# python vit.py --coef_est_type weiner --exp_name tff_vitFT_4 --wbits 4 --dataset inet1k_birds --dataset_dir data/inet1k_classes/birds \
#  --ckpt_path output/olvi1_outputs/inet1k_birds-2023-10-02-22-25-22/inet1k_birds_final_ckpt.bin --parent_dir birds_vitFT 
# python vit.py --coef_est_type weiner --exp_name tff_vitFT_3 --wbits 3 --dataset inet1k_birds --dataset_dir data/inet1k_classes/birds \
#  --ckpt_path output/olvi1_outputs/inet1k_birds-2023-10-02-22-25-22/inet1k_birds_final_ckpt.bin --parent_dir birds_vitFT 
# python vit.py --coef_est_type weiner --exp_name tff_vitFT_2 --wbits 2 --dataset inet1k_birds --dataset_dir data/inet1k_classes/birds \
#  --ckpt_path output/olvi1_outputs/inet1k_birds-2023-10-02-22-25-22/inet1k_birds_final_ckpt.bin --parent_dir birds_vitFT 

# python vit.py --coef_est_type weiner --exp_name fullW_tff_vitFT_16 --wbits 16 --dataset inet1k_trucks --dataset_dir data/inet1k_classes/trucks\
#  --ckpt_path output/olvi1_outputs/inet1k_trucks-2023-09-24-20-47-28/inet1k_trucks_final_ckpt.bin --parent_dir trucks_vitFT 
# python vit.py --coef_est_type weiner --exp_name fullW_tff_vitFT_4 --wbits 4 --dataset inet1k_trucks --dataset_dir data/inet1k_classes/trucks\
#  --ckpt_path output/olvi1_outputs/inet1k_trucks-2023-09-24-20-47-28/inet1k_trucks_final_ckpt.bin --parent_dir trucks_vitFT 
# python vit.py --coef_est_type weiner --exp_name fullW_tff_vitFT_3 --wbits 3 --dataset inet1k_trucks --dataset_dir data/inet1k_classes/trucks\
#  --ckpt_path output/olvi1_outputs/inet1k_trucks-2023-09-24-20-47-28/inet1k_trucks_final_ckpt.bin --parent_dir trucks_vitFT 
# python vit.py --coef_est_type weiner --exp_name fullW_tff_vitFT_2 --wbits 2 --dataset inet1k_trucks --dataset_dir data/inet1k_classes/trucks\
#  --ckpt_path output/olvi1_outputs/inet1k_trucks-2023-09-24-20-47-28/inet1k_trucks_final_ckpt.bin --parent_dir trucks_vitFT 

# python vit.py --coef_est_type weiner --exp_name gptq_vitFT_16 --wbits 16 --dataset inet1k_trucks --dataset_dir data/inet1k_classes/trucks\
#  --ckpt_path output/olvi1_outputs/inet1k_trucks-2023-09-24-20-47-28/inet1k_trucks_final_ckpt.bin --parent_dir trucks_vitFT 
# python vit.py --coef_est_type weiner --exp_name gptq_vitFT_4 --wbits 4 --dataset inet1k_trucks --dataset_dir data/inet1k_classes/trucks\
#  --ckpt_path output/olvi1_outputs/inet1k_trucks-2023-09-24-20-47-28/inet1k_trucks_final_ckpt.bin --parent_dir trucks_vitFT 
# python vit.py --coef_est_type weiner --exp_name gptq_vitFT_3 --wbits 3 --dataset inet1k_trucks --dataset_dir data/inet1k_classes/trucks\
#  --ckpt_path output/olvi1_outputs/inet1k_trucks-2023-09-24-20-47-28/inet1k_trucks_final_ckpt.bin --parent_dir trucks_vitFT 
# python vit.py --coef_est_type weiner --exp_name gptq_vitFT_2 --wbits 2 --dataset inet1k_trucks --dataset_dir data/inet1k_classes/trucks\
#  --ckpt_path output/olvi1_outputs/inet1k_trucks-2023-09-24-20-47-28/inet1k_trucks_final_ckpt.bin --parent_dir trucks_vitFT 

# python vit.py --coef_est_type weiner --exp_name tff_vitFT_16 --wbits 16 --dataset inet1k_trucks --dataset_dir data/inet1k_classes/trucks\
#  --ckpt_path output/olvi1_outputs/inet1k_trucks-2023-09-24-20-47-28/inet1k_trucks_final_ckpt.bin --parent_dir trucks_vitFT 
# python vit.py --coef_est_type weiner --exp_name tff_vitFT_4 --wbits 4 --dataset inet1k_trucks --dataset_dir data/inet1k_classes/trucks\
#  --ckpt_path output/olvi1_outputs/inet1k_trucks-2023-09-24-20-47-28/inet1k_trucks_final_ckpt.bin --parent_dir trucks_vitFT 
# python vit.py --coef_est_type weiner --exp_name tff_vitFT_3 --wbits 3 --dataset inet1k_trucks --dataset_dir data/inet1k_classes/trucks\
#  --ckpt_path output/olvi1_outputs/inet1k_trucks-2023-09-24-20-47-28/inet1k_trucks_final_ckpt.bin --parent_dir trucks_vitFT 
# python vit.py --coef_est_type weiner --exp_name tff_vitFT_2 --wbits 2 --dataset inet1k_trucks --dataset_dir data/inet1k_classes/trucks\
#  --ckpt_path output/olvi1_outputs/inet1k_trucks-2023-09-24-20-47-28/inet1k_trucks_final_ckpt.bin --parent_dir trucks_vitFT 

