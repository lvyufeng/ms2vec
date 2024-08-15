# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Create CoSENT model for text matching task
"""
import mindspore
from mindnlp.core import ops

from .sentence_model import SentenceModel

class CosentModel(SentenceModel):
    def __init__(
            self,
            model_name_or_path: str = "hfl/chinese-macbert-base",
            encoder_type: str = "FIRST_LAST_AVG",
            max_seq_length: int = 128,
            device: str = None,
    ):
        """
        Initializes a CoSENT Model.

        Args:
            model_name_or_path: Default Transformer model name or path to a directory containing Transformer model file (pytorch_nodel.bin).
            encoder_type: Enum of type EncoderType.
            max_seq_length: The maximum total input sequence length after tokenization.
            device: The device on which the model is allocated.
        """
        super().__init__(model_name_or_path, encoder_type, max_seq_length, device)

    def __str__(self):
        return f"<CoSENTModel: {self.model_name_or_path}, encoder_type: {self.encoder_type}, " \
               f"max_seq_length: {self.max_seq_length}>"

    def calc_loss(self, y_true, y_pred):
        """
        矩阵计算batch内的cos loss
        """
        # 1. 取出真实的标签
        y_true = y_true[::2]  # tensor([1, 0, 1]) 真实的标签
        # 2. 对输出的句子向量进行l2归一化   后面只需要对应为相乘  就可以得到cos值了
        norms = ops.sum((y_pred ** 2), dim=1, keepdim=True) ** 0.5
        y_pred = y_pred / norms
        # 3. 奇偶向量相乘, 相似度矩阵除以温度系数0.05(等于*20)
        y_pred = ops.sum(y_pred[::2] * y_pred[1::2], dim=1) * 20
        # 4. 取出负例-正例的差值
        y_pred = y_pred[:, None] - y_pred[None, :]  # 这里是算出所有位置 两两之间余弦的差值
        # 矩阵中的第i行j列  表示的是第i个余弦值-第j个余弦值
        y_true = y_true[:, None] < y_true[None, :]  # 取出负例-正例的差值
        y_true = y_true.float()
        y_pred = y_pred - (1 - y_true) * 1e12
        y_pred = y_pred.view(-1)
        # 这里加0是因为e^0 = 1相当于在log中加了1
        y_pred = ops.cat((mindspore.tensor([0]).float(), y_pred), dim=0)
        return ops.logsumexp(y_pred, dim=0)
