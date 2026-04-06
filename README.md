<div align="center">
<h2 align="center">
<b>PTQ1.61: Push the Real Limit of Extremely Low-Bit Post-Training Quantization Methods for Large Language Models </b>
</h2>
<div>
<a target="_blank" href="https://scholar.google.com/citations?user=lrJ0VWYAAAAJ&hl=zh-CN">Jiaqi&#160;Zhao</a><sup>1</sup>,
<a target="_blank" href="https://scholar.google.com/citations?user=6EUV_UMAAAAJ&hl=en">Miao&#160;Zhang</a><sup>1&#9993</sup>,
<a target="_blank" href="https://ieeexplore.ieee.org/author/37086659842">Ming&#160;Wang</a><sup>1</sup>,
<a target="_blank" href="https://scholar.google.com/citations?user=6ZPL5E0AAAAJ&hl=zh-CN">Yuzhang&#160;Shang</a><sup>2</sup>,
<a target="_blank" href="https://scholar.google.com/citations?user=eqwDXdMAAAAJ&hl=en">Kaihaoi&#160;Zhang</a><sup>1</sup>,
<a target="_blank" href="https://ieeexplore.ieee.org/author/37087008154">Weili&#160;Guan</a><sup>1</sup>,
 <br>
<a target="_blank" href="https://scholar.google.com/citations?user=o_DllmIAAAAJ&hl=zh-CN">Yaowei&#160;Wang</a><sup>2</sup>,
<a target="_blank" href="https://scholar.google.com/citations?user=CncXH-YAAAAJ&hl=en">Min&#160;Zhang</a><sup>1</sup>
</div>
<sup>1</sup>Harbin Institute of Technology, Shenzhen&#160&#160&#160</span>
<sup>2</sup>Illinois Institute of Technology</span>
<br />
<sup>&#9993&#160;</sup>Corresponding author&#160;&#160;</span>
<br/>
<div align="center">
    <a href="https://aclanthology.org/2025.acl-long.225.pdf" target="_blank">
    <img src="https://img.shields.io/badge/Paper-arXiv-deepgreen" alt="Paper arXiv"></a>
</div>
</div>

Large Language Models (LLMs) suffer severe performance degradation when facing extremely low-bit (sub 2-bit) quantization. Several existing sub 2-bit post-training quantization (PTQ) methods utilize a mix-precision scheme by leveraging an unstructured fine-grained mask to explicitly distinguish salient weights, while which introduces an extra 1-bit or more per weight. To explore the real limit of PTQ, we propose an extremely low-bit PTQ method called PTQ1.61, which enables weight quantization to 1.61-bit for the first time. Specifically, we first introduce a one-dimensional structured mask with negligibly additional 0.0002-bit per weight based on input activations from the perspective of reducing the upper bound of quantization error to allocate corresponding salient weight channels to 4-bit. For non-salient channels binarization, an efficient block-wise scaling factors optimization framework is then presented to take implicit row-wise correlations and angular biases into account. Different from prior works that concentrate on adjusting quantization methodologies, we further propose a novel paradigm called quantization preprocessing, where we argue that transforming the weight distribution of the pretrained model before quantization can alleviate the difficulty in per-channel extremely low-bit PTQ. Extensive experiments indicate our PTQ1.61 achieves state-of-the-art performance in extremely low-bit quantization. 

## Block-wise Optimization and Evaluation
We use LLaMa-7B as an example here:
1. Obtain the channel-wise scales required for initialization:
```
python generate_act_scale_shift.py --model /PATH/TO/LLaMA/llama-7b
```

2. Training and Evaluating PPL
```
CUDA_VISIBLE_DEVICES=0 python main.py --model /PATH/TO/LLAMA/llama-7b --epochs 20 --output_dir ./log/llama-7b --eval_ppl --wbits 4 --abits 16 --quant_type mix --lwc \
--save_dir /CHECKPOINT/TO/FIRST/PTQ \
--calib_dataset wikitext2  \
```

More detailed and optional arguments:
- `--model`: the local model path or huggingface format.
- `--wbits`: weight quantization bits.
- `--quant_type`: quantization type, mix means using structured masks.
- `--lwc`: activate the weight quantizer.
- `--epochs`: training epochs.
- `--nsamples`: number of calibration samples, 128 as default.
- `--eval_ppl`: evaluating the perplexity of quantized models.
- `--multigpu`: to inference larger network on multiple GPUs
- `--save_dir`: saving the quantization model for further exploration.

3. Reproduce the evaluation results of our paper.

1) Download the prebuilt quantized model from our anonymous huggingface repo: https://huggingface.co/ptq161.
2) The detailed reproduction methods please refer to **reproduce.ipynb**.


## Quantization Preprocessing
3. Preprocessing

After the first time PTQ, run:
```
cd preprocessing
CUDA_VISIBLE_DEVICES=0 python restorative_lora.py --model_id /PATH/TO/LLAMA/llama-7b \
--ckpt /CHECKPOINT/TO/FIRST/PTQ

CUDA_VISIBLE_DEVICES=0 python test_perplexity.py  --model_path /PATH/TO/LLAMA/llama-7b \
--ckpt /CHECKPOINT/TO/FIRST/PTQ \
--lora_path ./outputs/CHECKPOINT_NAME/step-r \
--output_path /PATH/TO/MERGED/MODEL
```
4. Evaluation PPL after Preprocessing
```
CUDA_VISIBLE_DEVICES=0 python main.py --model /PATH/TO/MERGED/MODEL --epochs 20 --output_dir ./log/llama-7b --eval_ppl --wbits 4 --abits 16 --quant_type mix --lwc \
--save_dir /CHECKPOINT/TO/SECOND/PTQ \
--calib_dataset wikitext2  \
```

## Reasoning Tasks Evaluation

Please follow lm-eval-harness for evaluating Hellaswag, PIQA, MMLU, GSM8K, LAMBADA, etc. 

'lm_eval' file is lm-evaluation-harness, a open-sourced evaluation framework from https://github.com/EleutherAI/lm-evaluation-harness, contains datasets, benchmarks, etc.
