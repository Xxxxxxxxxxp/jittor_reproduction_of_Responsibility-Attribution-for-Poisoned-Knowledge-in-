# jittor_reproduction_of_Responsibility-Attribution-for-Poisoned-Knowledge-in-RAG

本项目来自新芽计划，是对论文《Who Taught the Lie? Responsibility Attribution for Poisoned Knowledge in Retrieval-Augmented Generation》的复现分别使用pytorch，jittor框架进行复现
## 论文简介
论文提出了 RAGOrigin，这是一种黑盒归因框架 。这意味着它不需要访问 LLM 的内部参数或梯度，仅通过观察系统的输入和输出来识别中毒文档
核心机制：黑盒归因：利用相似度得分（similarity score）等指标来评估文档与生成内容之间的关联；动态环境适应性：RAGOrigin 被设计为可以在动态变化的知识库中有效工作;清理与防御：一旦识别出中毒文档，系统可以将其移除，从而修复 RAG 系统的知识库并防止未来的错误生成
## 环境配置
1.请通过以下指令配置环境
```bash
conda env create my_custom_env python=3.11
conda activate my_custom_env
pip install -r requirements.txt
```
2.请导入自己的密钥
```bash
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
export OPENAI_API_URL="YOUR_OPENAI_BASE_URL"
```
3.配置环境时，请尽量避免jittor与pytorch混装，以防numpy等库和两种框架的兼容性问题
## 攻击数据收集
为了确保本次复现实验具有高度的可追溯性与横向可比性，以及考虑到知识中毒攻击（Knowledge Poisoning）样本具有一定的敏感性，我们严格遵循了原论文所采用的公开的标准基准数据集(即dataset "NQ"，attack_retriever "e5"attack_LLM "gpt-4o-mini" ，judge_LLM "gpt-4o-mini" ，attack_method "PRAGB" ，attack_M 5 ，top_K 5 )。通过引用已验证的公开攻击样本，旨在消除数据分布差异对实验结果带来的随机干扰，从而在统一的基准线上更客观地验证 RAGOrigin 框架在责任归因方面的核心效能

## 最终复现结果测试
(在复现中的proxy_model中为了提高复现效率并适配主流科研环境的硬件约束，本项目使用轻量化复现方案。考虑到8B级别模型对显存的高要求，我们通过使用3B，1B等模型替代了原论文中的大参数代理模型。实验证明，该替代方案在保持归因准确率（DACC）的同时，显著降低了推理延迟和算力准入门槛)
\\
实验结果 1，在judge_LLM不变的情况(恒为gpt-4-mini)下改变proxy_LLM
| DACC | Llama-3.2-1B-Instruct |Llama-3.2-3B-Instruct|Qwen2.5-0.5B-Instruct|Qwen2.5-1.5B-Instruct|Qwen2.5-3B-Instruct|
|--------|--------|--------|--------|---------|-----------|
| Pytorch  | 0.9668508287292817  | 0.9651933701657458  |0.9936046511627907|0.998|0.9975460122699387|
| Jittor | 0.99  |0.993  |0.991|0.995|0.986|
| 作者原代码 | 0.99  | 1.00  |0.99|0.99|0.99|


| FPR | Llama-3.2-1B-Instruct |Llama-3.2-3B-Instruct|Qwen2.5-0.5B-Instruct|Qwen2.5-1.5B-Instruct|Qwen2.5-3B-Instruct|
|--------|--------|--------|--------|---------|----------|
| Pytorch  | 0.04580152671755725  | 0.048091603053435114  |0.00819672131147541|0.004|0.0035398230088495575|
| Jittor | 0.02  | 0.014 |0.018|0.001|0.028|
| 作者原代码 | 0.01  | 0.0  |0.01|0.01|0.01|



| FNR | Llama-3.2-1B-Instruct |Llama-3.2-3B-Instruct|Qwen2.5-0.5B-Instruct|Qwen2.5-1.5B-Instruct|Qwen2.5-3B-Instruct|
|--------|--------|--------|----------|----------|---------|
| Pytorch  | 0.0  |0.0|0.002|0.0 | 0.0|
| Jittor | 0.0  | 0.0  |0.0| 0.0|  0.0|
| 作者原代码 | 0.0  | 0.0  |0.0|0.0|0.0|

实验结果2.在相同proxy_LLM的情况下(恒为Llama-3.2-1B-Instruct)改变judge_LLM
| DACC | gpt-3.5-turbo |gpt-4o|gpt-4o-mini|deepseek-V3|Llama-3.2-8B-Instruct|llama-3-70b-instruct|qwen2.5-7b-instruct|qwen2.5-32b-instruct|qwen2.5-72b-instruct|
|--------|--------|--------|--------|---------|-----------|-------------|------------|--------|-----------|
| Pytorch  |0.9873684210526316|0.9492822966507177|0.9805194805194806|0.9869346733668342|0.9807692307692307|0.9791304347826087|0.9811242337267|0.98|0.988|
| Jittor | 0.99  |0.99  |0.99|0.995|0.99|0.99|0.992|0.991|0.988|
| 作者原代码 | 1.00  | 1.00  |0.99|1.00|1.00|1.00|1.00|1.00|1.00|


| FPR | gpt-3.5-turbo |gpt-4o|gpt-4o-mini|deepseek-V3|Llama-3.2-8B-Instruct|llama-3-70b-instruct|qwen2.5-7b-instruct|qwen2.5-32b-instruct|qwen2.5-72b-instruct|
|--------|--------|--------|--------|---------|-----------|-------------|------------|--------|-----------|
| Pytorch  | 0.017142857142857144  |  0.057608695652173914  |0.028846153846153848|0.0174496644295302|0.02531645569620253|0.026667|0.04435204421|0.04|0.04|
| Jittor | 0.2  |0.02  |0.02|0.02|0.02|0.02|0.2|0.198|0.231|
| 作者原代码 | 0.0  | 0.0 |0.01|0.0|0.0|0.0|0.0|0.0|0.0|


| FNR | gpt-3.5-turbo |gpt-4o|gpt-4o-mini|deepseek-V3|Llama-3.2-8B-Instruct|llama-3-70b-instruct|qwen2.5-7b-instruct|qwen2.5-32b-instruct|qwen2.5-72b-instruct|
|--------|--------|--------|--------|---------|-----------|-------------|------------|--------|-----------|
| Pytorch  | 0.0  |  0.0 |0.0|0.0|0.0|0.0|0.0|0.0|0.0|
| Jittor | 0.0 |0.0  |0.0|0.0|0.0||0.0|0.0|0.0|
| 作者原代码 |0.0  | 0.0  |0.0|0.0|0.0|0.0|0.0|0.0|0.0|


