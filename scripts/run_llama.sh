# # gptq
python llama.py /data/harsha/llama/hf_llama2/ c4 --eval --exp_name llama_gptq_2 --new_eval --wbits 2 --quant gptq --pre_gptqH --qfn a 
# 
# # quip
python llama.py /data/harsha/llama/hf_llama2/ c4 --eval --exp_name llama_quip_2 --new_eval --wbits 2 --quant ldlq --pre_gptqH --pre_proj --qfn b 
# 
# # frameQuant
# python llama.py /data/harsha/llama/hf_llama2/ c4 --eval --exp_name llama_fq_2 --new_eval --wbits 2 --quant ldlq --pre_gptqH --pre_tff --qfn s --x_sigma 2.0 --pre_proj 
# 
# # full precision
python llama.py /data/harsha/llama/hf_llama2/ c4 --eval --exp_name llama_fp16 --new_eval --wbits 16
# 
# redundancy
python llama.py ./hf_llama2/ c4 --eval --exp_name llama_fq_2   --tff_redundancy 1.0 --new_eval --wbits 2 --quant ldlq --pre_gptqH --pre_tff --qfn s --x_sigma 2.0 --pre_proj 
python llama.py ./hf_llama2/ c4 --eval --exp_name llama_fq_2p2 --tff_redundancy 1.1 --new_eval --wbits 2 --quant ldlq --pre_gptqH --pre_tff --qfn s --x_sigma 2.0 --pre_proj 
python llama.py ./hf_llama2/ c4 --eval --exp_name llama_fq_2p4 --tff_redundancy 1.2 --new_eval --wbits 2 --quant ldlq --pre_gptqH --pre_tff --qfn s --x_sigma 2.0 --pre_proj 
python llama.py ./hf_llama2/ c4 --eval --exp_name llama_fq_2p6 --tff_redundancy 1.3 --new_eval --wbits 2 --quant ldlq --pre_gptqH --pre_tff --qfn s --x_sigma 2.0 --pre_proj 
python llama.py ./hf_llama2/ c4 --eval --exp_name llama_fq_2p8 --tff_redundancy 1.4 --new_eval --wbits 2 --quant ldlq --pre_gptqH --pre_tff --qfn s --x_sigma 2.0 --pre_proj 

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
