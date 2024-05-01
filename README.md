# ms2vec: MindSpore Text to Vector

移植自shibing624的[text2vec](https://github.com/shibing624/text2vec)库。

**Text2vec**: Text to Vector, Get Sentence Embeddings. 文本向量化，把文本(包括词、句子、段落)表征为向量矩阵。

**text2vec**实现了Word2Vec、RankBM25、BERT、Sentence-BERT、CoSENT等多种文本表征、文本相似度计算模型，并在文本语义匹配（相似度计算）任务上比较了各模型的效果。

**Guide**
- [Features](#Features)
- [Evaluation](#Evaluation)
- [Install](#install)
- [Usage](#usage)
- [Contact](#Contact)
- [References](#references)


## Features
### 文本向量表示模型
- [Word2Vec](https://github.com/shibing624/text2vec/blob/master/text2vec/word2vec.py)：通过腾讯AI Lab开源的大规模高质量中文[词向量数据（800万中文词轻量版）](https://pan.baidu.com/s/1La4U4XNFe8s5BJqxPQpeiQ) (文件名：light_Tencent_AILab_ChineseEmbedding.bin 密码: tawe）实现词向量检索，本项目实现了句子（词向量求平均）的word2vec向量表示
- [SBERT(Sentence-BERT)](https://github.com/shibing624/text2vec/blob/master/text2vec/sentencebert_model.py)：权衡性能和效率的句向量表示模型，训练时通过有监督训练BERT和softmax分类函数，文本匹配预测时直接取句子向量做余弦，句子表征方法，本项目基于MindSpore复现了Sentence-BERT模型的预测
- [CoSENT(Cosine Sentence)](https://github.com/shibing624/text2vec/blob/master/text2vec/cosent_model.py)：CoSENT模型提出了一种排序的损失函数，使训练过程更贴近预测，模型收敛速度和效果比Sentence-BERT更好，本项目基于MindSpore实现了CoSENT模型的预测
- [BGE(BAAI general embedding)](https://github.com/shibing624/text2vec/blob/master/text2vec/bge_model.py)：BGE本项目基于MindSpore实现了BGE模型的预测


详细文本向量表示方法见wiki: [文本向量表示方法](https://github.com/shibing624/text2vec/wiki/%E6%96%87%E6%9C%AC%E5%90%91%E9%87%8F%E8%A1%A8%E7%A4%BA%E6%96%B9%E6%B3%95)

