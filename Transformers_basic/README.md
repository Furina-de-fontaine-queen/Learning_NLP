# README

## MyTransformers - A Simplified Transformer Implementation  
## MyTransformers - 简化版Transformer实现

### Overview  
### 概述

This repository contains a Python implementation of the Transformer model architecture as described in the seminal paper "Attention Is All You Need". The code is designed to be educational, focusing on clarity and understanding of the core concepts rather than production-level performance.  
本仓库包含基于经典论文《Attention Is All You Need》实现的Transformer模型架构。代码设计以教学为目的，侧重于核心概念的清晰呈现和理解，而非生产级性能。

### Current Implementation  
### 当前实现

The code currently implements the following key components of the Transformer architecture:  
当前代码实现了Transformer架构的以下核心组件：

1. **Core Modules**:  
   **核心模块**:
   - Multi-head attention mechanism  
   - 多头注意力机制
   - Position-wise feed-forward networks  
   - 位置前馈网络
   - Positional encoding  
   - 位置编码
   - Layer normalization  
   - 层归一化
   - Residual connections  
   - 残差连接

2. **Main Architecture**:  
   **主要架构**:
   - Encoder and Decoder stacks  
   - 编码器和解码器堆栈
   - Encoder-Decoder structure  
   - 编码器-解码器结构
   - Embedding layers (token + positional)  
   - 嵌入层(词元+位置)

3. **Training Utilities**:  
   **训练工具**:
   - Label smoothing  
   - 标签平滑
   - Noam optimizer (learning rate scheduling)  
   - Noam优化器(学习率调度)
   - Batch processing  
   - 批处理
   - Masking utilities  
   - 掩码工具

4. **Supporting Components**:  
   **支持组件**:
   - Attention masking (padding and sequence)  
   - 注意力掩码(填充和序列)
   - Sub-layer connections  
   - 子层连接
   - Layer normalization  
   - 层归一化
   - Embedding layers  
   - 嵌入层

### Future Plans  
### 未来计划

This is an ongoing learning project. Future updates will include:  
这是一个持续学习项目，未来更新将包含：

1. Deeper exploration of the mathematical foundations of Transformers  
1. 深入探索Transformer的数学基础
2. More detailed documentation of the architecture and design choices  
2. 更详细的架构和设计选择文档
3. Additional transformer variants and improvements  
3. 更多Transformer变体和改进
4. Practical applications and examples  
4. 实际应用案例
5. Performance optimizations  
5. 性能优化

### References  
### 参考文献

The implementation is primarily based on:  
实现主要基于：

- The original "Attention Is All You Need" paper by Vaswani et al.  
- Vaswani等人的原始论文《Attention Is All You Need》
- Various educational blog posts and articles about Transformers  
- 关于Transformer的各种教育博客和文章
- Online tutorials and video explanations  
- 在线教程和视频讲解
- Other open-source implementations  
- 其他开源实现

### Usage  
### 使用说明

The code is designed to be modular and can be used as:  
代码采用模块化设计，可用作：

- A learning reference for understanding Transformers  
- 理解Transformer的学习参考
- A base for building custom transformer models  
- 构建自定义Transformer模型的基础
- A starting point for experimentation  
- 实验的起点

### Contribution  
### 贡献指南

Contributions and suggestions are welcome! Please open issues or pull requests for any improvements or corrections.  
欢迎贡献和建议！如有改进或修正，请提交issue或pull request。
