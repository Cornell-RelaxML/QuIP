
# FP
python opt.py facebook/opt-125m c4 --exp_name opt125m_fp16 --wbits 16 
# GPTQ
python opt.py facebook/opt-125m c4 --exp_name opt125m_gptq4 --tff_redundancy 1.0 --wbits 2 --quant ldlq --pre_gptqH --pre_tff --qfn s --x_sigma 2.0 --pre_proj --emb_quant gptq --emb_wbits 4 --emb_qfn a --emb_quant_en --emb_pre_gptqH 
python opt.py facebook/opt-125m c4 --exp_name opt125m_gptq2 --tff_redundancy 1.0 --wbits 2 --quant ldlq --pre_gptqH --pre_tff --qfn s --x_sigma 2.0 --pre_proj --emb_quant gptq --emb_wbits 2 --emb_qfn a --emb_quant_en --emb_pre_gptqH
# FrameQuant
python opt.py facebook/opt-125m c4 --exp_name opt125m_fq4 --tff_redundancy 1.0 --wbits 2 --quant ldlq --pre_gptqH --pre_tff --qfn s --x_sigma 2.0 --pre_proj --emb_quant ldlq --emb_wbits 4 --emb_qfn s --emb_quant_en --emb_pre_gptqH --emb_pre_proj 
python opt.py facebook/opt-125m c4 --exp_name opt125m_fq2 --tff_redundancy 1.0 --wbits 2 --quant ldlq --pre_gptqH --pre_tff --qfn s --x_sigma 2.0 --pre_proj --emb_quant ldlq --emb_wbits 2 --emb_qfn s --emb_quant_en --emb_pre_gptqH --emb_pre_proj
# Nearest
python opt.py facebook/opt-125m c4 --exp_name opt125m_near4 --tff_redundancy 1.0 --wbits 2 --quant ldlq --pre_gptqH --pre_tff --qfn s --x_sigma 2.0 --pre_proj --emb_quant nearest --emb_wbits 4 --emb_qfn a --emb_quant_en 
python opt.py facebook/opt-125m c4 --exp_name opt125m_near2 --tff_redundancy 1.0 --wbits 2 --quant ldlq --pre_gptqH --pre_tff --qfn s --x_sigma 2.0 --pre_proj --emb_quant nearest --emb_wbits 2 --emb_qfn a --emb_quant_en 