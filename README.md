# jittor_reproduction_of_Responsibility-Attribution-for-Poisoned-Knowledge-in-RAG

本项目来自新芽计划，是对论文《Who Taught the Lie? Responsibility Attribution for Poisoned Knowledge in Retrieval-Augmented Generation》的复现分别使用pytorch，jittor框架进行复现
## 论文简介
论文提出了 RAGOrigin，这是一种黑盒归因框架 。这意味着它不需要访问 LLM 的内部参数或梯度，仅通过观察系统的输入和输出来识别中毒文档
核心机制：黑盒归因：利用相似度得分（similarity score）等指标来评估文档与生成内容之间的关联；动态环境适应性：RAGOrigin 被设计为可以在动态变化的知识库中有效工作;清理与防御：一旦识别出中毒文档，系统可以将其移除，从而修复 RAG 系统的知识库并防止未来的错误生成
## 环境配置
1.请运行以下代码配置环境
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
## 攻击数据收集
为了确保本次复现实验具有高度的可追溯性与横向可比性，以及考虑到知识中毒攻击（Knowledge Poisoning）样本具有一定的敏感性，我们严格遵循了原论文所采用的标准基准数据集。通过引用已验证的公开攻击样本，旨在消除数据分布差异对实验结果带来的随机干扰，从而在统一的基准线上更客观地验证 RAGOrigin 框架在责任归因方面的核心效能

## 
