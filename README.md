# QuIP: Quantization with Incoherence Processing

This repository contains code for the paper [**QuIP: 2-Bit Quantization of Large Language Models with Guarantees**](https://arxiv.org/pdf/2307.13304.pdf). 

**TLDR:** Our proposed incoherence processing enables quantization of large language models down to 2 bits.
Please see our paper for full details.

The code is built on top of [OPTQ's repository](https://github.com/IST-DASLab/gptq). The current code includes the following: 


## Update: [QuIP#](https://github.com/Cornell-RelaxML/quip-sharp) is our new and improved method! Includes a lattice codebook and an efficient cuda implementation! Results on quantizing Llama 1 and 2 models, achieving near fp16 quantization performance at 2 bits. 

### Llama-2 Update
Replace `opt.py` with `llama.py` to quantize and evaluate the Llama-2 class of models with QuIP. 
Note that we currently evaluate this model with 2048 context length, but this can be changed by modifying `model.seqlen`.

## Language Generation

```
# Compute full precision (FP16) results
CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-125m c4
# Run a quantization method with Incoherence Processing
CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-125m c4 --wbits 4 --quant <quantmethod> --incoh_processing --save <savename>
# Run a quantization method with baseline processing
CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-125m c4 --wbits 4 --quant gptq --pre_gptqH --save <savename>
````

Quantization methods include:
- `ldlq`: runs the LDLQ rounding algorithm (we show its equivalence to OPTQ, providing a novel theoretical analysis)
- `ldlqRG`: runs the LDLQ_RG algorithm with additional hessian-based hessian reordering, and further greedy updates, with `--npasses` controlling the number of passes over the weights
- `gptq`: runs OPTQ algorithm as implemented by its authors
- `allbal`: algorithm to run greedy updates by themselves, with `--npasses` the argument controlling the number of passes over the weights
- `ldlbal_admm`: alternative algorithm which constraints the rounded weights to be sufficiently close to their original, giving a better theoretical bound.

The `--incoh_processing` argument is a meta argument which sets the following flags `--pre_gptqH --pre_rescale --pre_proj --qfn b`. 
For more control into the pre and post processing, these arguments can be set individually.

To run other OPT models replace `opt-125m` with one of: `opt-350m`, `opt-1.3b`, `opt-2.7b`, `opt-6.7b`, `opt-13b`, `opt-30b`, etc.
On larger models, a low compute-to-memory-access ratio can slow down the quantization algorithms. 
We implement a lazy batch update to te weight matrix specified by `--lazy_batch`.
This argument works with the quantization methods {ldlq, ldlqRG, allbal}.
Note OPTQ already implements this, and is where we got the idea from.

## ZeroShot

```
# Compute full precision (FP16) results 
CUDA_VISIBLE_DEVICES=0 python main.py facebook/opt-125m c4 --wbits 16 --nsamples 0 --task <task>
# Evaluate saved model
CUDA_VISIBLE_DEVICES=0 python main.py facebook/opt-125m c4 --load <load_name> --nsamples 0 --task <task>
```
To evaluate the quantized models on zeroshot tasks, simply provide the saved quantized model weights to the script.
Evaluated tasks are {arc_easy, lambada, piqa, storycloze}.

## Benchmarking
Please see our new project QuIP#.


## OPTQ and LDLQ Equivalence
Run the following script to empirically verify that the output of OPTQ's implementation and our implementation of LDLQ are identical: `python optq_ldlq_equiv.py`.
Note OPTQ's implementation requires running on a GPU.

## OTPQ/LDLQ Finite Grid Counterexample
Run `python optq_counter.py` to compute the proxy loss of our W,H counterexample. 

## Computing Proxy Loss
In a similar manner to `opt.py`, run `opt_saveH.py` to save the H matrices resulting from the specified model and quantization method.
Then, run `opt_proxy.py` to compute the proxy loss for a specified quantization method. 
```
CUDA_VISIBLE_DEVICES=0 python opt_proxy.py c4 --wbits 4 --quant <quant_method>
```

## H Summary
Run the following script to compute summary statistics of a folder `<dirname>` of H matrices, output from running `opt_saveH.py`. 
```
python compute_Hsummary.py --dirname <> --savename <> 
```
