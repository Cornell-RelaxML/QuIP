# # gptq
# python llama.py /data/harsha/llama/hf_llama2/ c4 --eval --exp_name llama_gptq_2 --new_eval --wbits 2 --quant gptq --pre_gptqH --qfn a 
# 
# # quip
# python llama.py /data/harsha/llama/hf_llama2/ c4 --eval --exp_name llama_quip_2 --new_eval --wbits 2 --quant ldlq --pre_gptqH --pre_proj --qfn b 
# 
# # frameQuant
# python llama.py /data/harsha/llama/hf_llama2/ c4 --eval --exp_name llama_fq_2 --new_eval --wbits 2 --quant ldlq --pre_gptqH --pre_tff --qfn s --x_sigma 2.0 --pre_proj 
# 
# # full precision
# python llama.py /data/harsha/llama/hf_llama2/ c4 --eval --exp_name llama_fp16 --new_eval --wbits 16
# 
# redundancy

python opt.py facebook/opt-125m c4 --exp_name opt_125m_fq_2   --tff_redundancy 1.0 --wbits 2 --quant ldlq --pre_gptqH --pre_tff --qfn s --x_sigma 2.0 --pre_proj 
python opt.py facebook/opt-125m c4 --exp_name opt_125m_fq_2p2 --tff_redundancy 1.1 --wbits 2 --quant ldlq --pre_gptqH --pre_tff --qfn s --x_sigma 2.0 --pre_proj 
python opt.py facebook/opt-125m c4 --exp_name opt_125m_fq_2p4 --tff_redundancy 1.2 --wbits 2 --quant ldlq --pre_gptqH --pre_tff --qfn s --x_sigma 2.0 --pre_proj 
python opt.py facebook/opt-125m c4 --exp_name opt_125m_fq_2p6 --tff_redundancy 1.3 --wbits 2 --quant ldlq --pre_gptqH --pre_tff --qfn s --x_sigma 2.0 --pre_proj 
python opt.py facebook/opt-125m c4 --exp_name opt_125m_fq_2p8 --tff_redundancy 1.4 --wbits 2 --quant ldlq --pre_gptqH --pre_tff --qfn s --x_sigma 2.0 --pre_proj 

python opt.py facebook/opt-350m c4 --exp_name opt_350m_fq_2   --tff_redundancy 1.0 --wbits 2 --quant ldlq --pre_gptqH --pre_tff --qfn s --x_sigma 2.0 --pre_proj 
python opt.py facebook/opt-350m c4 --exp_name opt_350m_fq_2p2 --tff_redundancy 1.1 --wbits 2 --quant ldlq --pre_gptqH --pre_tff --qfn s --x_sigma 2.0 --pre_proj 
python opt.py facebook/opt-350m c4 --exp_name opt_350m_fq_2p4 --tff_redundancy 1.2 --wbits 2 --quant ldlq --pre_gptqH --pre_tff --qfn s --x_sigma 2.0 --pre_proj 
python opt.py facebook/opt-350m c4 --exp_name opt_350m_fq_2p6 --tff_redundancy 1.3 --wbits 2 --quant ldlq --pre_gptqH --pre_tff --qfn s --x_sigma 2.0 --pre_proj 
python opt.py facebook/opt-350m c4 --exp_name opt_350m_fq_2p8 --tff_redundancy 1.4 --wbits 2 --quant ldlq --pre_gptqH --pre_tff --qfn s --x_sigma 2.0 --pre_proj 

python opt.py facebook/opt-1.3b c4 --exp_name opt_1p3b_fq_2   --tff_redundancy 1.0 --wbits 2 --quant ldlq --pre_gptqH --pre_tff --qfn s --x_sigma 2.0 --pre_proj 
python opt.py facebook/opt-1.3b c4 --exp_name opt_1p3b_fq_2p2 --tff_redundancy 1.1 --wbits 2 --quant ldlq --pre_gptqH --pre_tff --qfn s --x_sigma 2.0 --pre_proj 
python opt.py facebook/opt-1.3b c4 --exp_name opt_1p3b_fq_2p4 --tff_redundancy 1.2 --wbits 2 --quant ldlq --pre_gptqH --pre_tff --qfn s --x_sigma 2.0 --pre_proj 
python opt.py facebook/opt-1.3b c4 --exp_name opt_1p3b_fq_2p6 --tff_redundancy 1.3 --wbits 2 --quant ldlq --pre_gptqH --pre_tff --qfn s --x_sigma 2.0 --pre_proj 
python opt.py facebook/opt-1.3b c4 --exp_name opt_1p3b_fq_2p8 --tff_redundancy 1.4 --wbits 2 --quant ldlq --pre_gptqH --pre_tff --qfn s --x_sigma 2.0 --pre_proj 

python opt.py facebook/opt-2.7b c4 --exp_name opt_2p7b_fq_2   --tff_redundancy 1.0 --wbits 2 --quant ldlq --pre_gptqH --pre_tff --qfn s --x_sigma 2.0 --pre_proj 
python opt.py facebook/opt-2.7b c4 --exp_name opt_2p7b_fq_2p2 --tff_redundancy 1.1 --wbits 2 --quant ldlq --pre_gptqH --pre_tff --qfn s --x_sigma 2.0 --pre_proj 
python opt.py facebook/opt-2.7b c4 --exp_name opt_2p7b_fq_2p4 --tff_redundancy 1.2 --wbits 2 --quant ldlq --pre_gptqH --pre_tff --qfn s --x_sigma 2.0 --pre_proj 
python opt.py facebook/opt-2.7b c4 --exp_name opt_2p7b_fq_2p6 --tff_redundancy 1.3 --wbits 2 --quant ldlq --pre_gptqH --pre_tff --qfn s --x_sigma 2.0 --pre_proj 
python opt.py facebook/opt-2.7b c4 --exp_name opt_2p7b_fq_2p8 --tff_redundancy 1.4 --wbits 2 --quant ldlq --pre_gptqH --pre_tff --qfn s --x_sigma 2.0 --pre_proj 

python opt.py facebook/opt-6.7b c4 --exp_name opt_6p7b_fq_2   --tff_redundancy 1.0 --wbits 2 --quant ldlq --pre_gptqH --pre_tff --qfn s --x_sigma 2.0 --pre_proj 
python opt.py facebook/opt-6.7b c4 --exp_name opt_6p7b_fq_2p2 --tff_redundancy 1.1 --wbits 2 --quant ldlq --pre_gptqH --pre_tff --qfn s --x_sigma 2.0 --pre_proj 
python opt.py facebook/opt-6.7b c4 --exp_name opt_6p7b_fq_2p4 --tff_redundancy 1.2 --wbits 2 --quant ldlq --pre_gptqH --pre_tff --qfn s --x_sigma 2.0 --pre_proj 
python opt.py facebook/opt-6.7b c4 --exp_name opt_6p7b_fq_2p6 --tff_redundancy 1.3 --wbits 2 --quant ldlq --pre_gptqH --pre_tff --qfn s --x_sigma 2.0 --pre_proj 
python opt.py facebook/opt-6.7b c4 --exp_name opt_6p7b_fq_2p8 --tff_redundancy 1.4 --wbits 2 --quant ldlq --pre_gptqH --pre_tff --qfn s --x_sigma 2.0 --pre_proj 

# # python llama.py /data/harsha/llama/hf_llama2/ c4 --eval --exp_name llama_fq_2p6 --tff_redundancy 1.3 --new_eval --wbits 2 --quant ldlq --pre_gptqH --pre_tff --qfn s --x_sigma 2.0 --pre_proj 

# full precision                            
# python llama.py /data/harsha/llama/hf_llama2_70b/ c4 --eval --exp_name llama_fp16_70b --new_eval --wbits 16
# 
# # gptq
# python llama.py /data/harsha/llama/hf_llama2_70b/ c4 --eval --exp_name llama_gptq_2_70b --new_eval --wbits 2 --quant gptq --pre_gptqH --qfn a 
#                                             
# # quip                                      
# python llama.py /data/harsha/llama/hf_llama2_70b/ c4 --eval --exp_name llama_quip_2_70b --new_eval --wbits 2 --quant ldlq --pre_gptqH --pre_proj --qfn b 
#                                             
# # frameQuant                                
# python llama.py /data/harsha/llama/hf_llama2_70b/ c4 --eval --exp_name llama_fq_2_70b --new_eval --wbits 2 --quant ldlq --pre_gptqH --pre_tff --qfn s --x_sigma 2.0 --pre_proj 
#                                             
# # redundancy                                
# python llama.py /data/harsha/llama/hf_llama2_70b/ c4 --eval --exp_name llama_fq_2_70b   --tff_redundancy 1.0 --new_eval --wbits 2 --quant ldlq --pre_gptqH --pre_tff --qfn s --x_sigma 2.0 --pre_proj 
# python llama.py /data/harsha/llama/hf_llama2_70b/ c4 --eval --exp_name llama_fq_2p2_70b --tff_redundancy 1.1 --new_eval --wbits 2 --quant ldlq --pre_gptqH --pre_tff --qfn s --x_sigma 2.0 --pre_proj 
# python llama.py /data/harsha/llama/hf_llama2_70b/ c4 --eval --exp_name llama_fq_2p4_70b --tff_redundancy 1.2 --new_eval --wbits 2 --quant ldlq --pre_gptqH --pre_tff --qfn s --x_sigma 2.0 --pre_proj 
