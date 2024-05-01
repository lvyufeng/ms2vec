# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Create Bge model for text matching task

code modified from https://github.com/FlagOpen/FlagEmbedding
"""

import mindspore
from mindspore import ops

from .sentence_model import SentenceModel



class BgeModel(SentenceModel):
    def __init__(
            self,
            model_name_or_path: str = "BAAI/bge-large-zh-noinstruct",
            encoder_type: str = "MEAN",
            max_seq_length: int = 32,
            passage_max_len: int = 128,
            device: str = None,
    ):
        """
        Initializes a Bge Model.

        Args:
            model_name_or_path: Default Transformer model name or path to a directory containing Transformer model file (pytorch_nodel.bin).
            encoder_type: encoder type, set by model name
            max_seq_length: The maximum total input sequence length after tokenization.
            passage_max_len: The maximum total input sequence length after tokenization.
            num_classes: Number of classes for classification.
            device: CPU or GPU
        """
        super().__init__(model_name_or_path, encoder_type, max_seq_length, device)
        self.query_max_len = max_seq_length
        self.passage_max_len = passage_max_len

    def __str__(self):
        return f"<BgeModel: {self.model_name_or_path}, encoder_type: {self.encoder_type}, " \
               f"max_seq_length: {self.max_seq_length}>"

    def calc_loss(self, y_true, y_pred):
        """
        Calc loss with two sentence embeddings, Softmax loss
        """
        loss = ops.cross_entropy(y_pred, y_true)
        return loss

    def calc_similarity(self, q_embs, p_embs):
        """
        Calc similarity with two sentence embeddings
        """
        if len(p_embs.shape) == 2:
            return ops.matmul(q_embs, p_embs.swapaxes(0, 1))
        return ops.matmul(q_embs, p_embs.swapaxes(-2, -1))

    @staticmethod
    def flat_list(l):
        return [item for sublist in l for item in sublist]
