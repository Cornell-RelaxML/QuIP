# FP
# python llama.py /data/harsha/llama/hf_llama2/ c4 --eval --exp_name llama7_fp16 --new_eval --wbits 16
# # FrameQuant (current) not quantize embedding tables and only quantize FC/Transformers
# python llama.py /data/harsha/llama/hf_llama2/ c4 --eval --exp_name llama7_fq2_emb   --tff_redundancy 1.0 --new_eval --wbits 2 --quant ldlq --pre_gptqH --pre_tff --qfn s --x_sigma 2.0 --pre_proj 
# # FrameQuant quantize embedding tables and quantize FC/Transformers
# python llama.py /data/harsha/llama/hf_llama2/ c4 --eval --exp_name llama7_fq2_emb   --tff_redundancy 1.0 --new_eval --wbits 2 --quant ldlq --pre_gptqH --pre_tff --qfn s --x_sigma 2.0 --pre_proj --emb_quant ldlq --emb_wbits 2 --emb_qfn s --emb_quant_en --emb_pre_gptqH --emb_pre_proj
# # FrameQuant quantize embedding tables, but not quantize FC/Transformers
# python llama.py /data/harsha/llama/hf_llama2/ c4 --eval --exp_name llama7_fq2_emb   --tff_redundancy 1.0 --new_eval --wbits 2 --quant ldlq --pre_gptqH --pre_tff --qfn s --x_sigma 2.0 --pre_proj --emb_quant ldlq --emb_wbits 2 --emb_qfn s --emb_quant_en --emb_pre_gptqH --emb_pre_proj --emb_quant_only

# FrameQuant quantize embedding tables and quantize FC/Transformers
python llama.py /data/harsha/llama/hf_llama2/ c4 --eval --exp_name llama7_fq42_et --tff_redundancy 1.0 --new_eval --wbits 2 --quant ldlq --pre_gptqH --pre_tff --qfn s --x_sigma 2.0 --pre_proj --emb_quant ldlq --emb_wbits 4 --emb_qfn s --emb_quant_en --emb_pre_gptqH --emb_pre_proj
# FrameQuant quantize embedding tables, but not quantize FC/Transformers
python llama.py /data/harsha/llama/hf_llama2/ c4 --eval --exp_name llama7_fq42_e  --tff_redundancy 1.0 --new_eval --wbits 2 --quant ldlq --pre_gptqH --pre_tff --qfn s --x_sigma 2.0 --pre_proj --emb_quant ldlq --emb_wbits 4 --emb_qfn s --emb_quant_en --emb_pre_gptqH --emb_pre_proj --emb_quant_only

# FrameQuant (current) not quantize embedding tables and only quantize FC/Transformers
python llama.py /data/harsha/llama/hf_llama2/ c4 --eval --exp_name llama7_fq44_t   --tff_redundancy 1.0 --new_eval --wbits 4 --quant ldlq --pre_gptqH --pre_tff --qfn s --x_sigma 2.0 --pre_proj 
# FrameQuant quantize embedding tables and quantize FC/Transformers
python llama.py /data/harsha/llama/hf_llama2/ c4 --eval --exp_name llama7_fq44_et   --tff_redundancy 1.0 --new_eval --wbits 4 --quant ldlq --pre_gptqH --pre_tff --qfn s --x_sigma 2.0 --pre_proj --emb_quant ldlq --emb_wbits 4 --emb_qfn s --emb_quant_en --emb_pre_gptqH --emb_pre_proj

# FrameQuant quantize embedding tables and quantize FC/Transformers
python llama.py /data/harsha/llama/hf_llama2/ c4 --eval --exp_name llama7_fq24_et   --tff_redundancy 1.0 --new_eval --wbits 2 --quant ldlq --pre_gptqH --pre_tff --qfn s --x_sigma 2.0 --pre_proj --emb_quant ldlq --emb_wbits 4 --emb_qfn s --emb_quant_en --emb_pre_gptqH --emb_pre_proj