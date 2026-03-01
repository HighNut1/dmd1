from abc import ABC, abstractmethod

import numpy as np

from flx.data.dataset import Identifier
from flx.data.embedding_loader import EmbeddingLoader


class BiometricMatcher(ABC):
    @abstractmethod
    def similarity(self, sample1: Identifier, sample2: Identifier) -> float:
        raise NotImplementedError()


class VectorizedMatcher(BiometricMatcher):
    @abstractmethod
    def preload_vectorized(self, samples: list[Identifier]) -> None:
        """
        Preloads all samples into one numpy ndarray for vectorized comparison.
        """
        raise NotImplementedError()

    @abstractmethod
    def vectorized_similarity(self, sample: Identifier) -> np.ndarray:
        """
        Similarities with all the samples in the preloaded vector.
        """
        raise NotImplementedError()


# 保持原类名不变，但内部逻辑修改为针对 {-1, 1} 二值化向量的汉明相似度计算
class CosineSimilarityMatcher(VectorizedMatcher):
    def __init__(self, embedding_dataset: EmbeddingLoader):
        self._embeddings = embedding_dataset
        self._matrix = None

    def similarity(self, sample1: Identifier, sample2: Identifier) -> float:
        emb1 = self._embeddings.get(sample1)
        emb2 = self._embeddings.get(sample2)
        
        # --- 修改点 1：使用汉明相似度映射 ---
        # 兼容原来的 .vector 属性或直接数组
        v1 = emb1.vector if hasattr(emb1, 'vector') else emb1
        v2 = emb2.vector if hasattr(emb2, 'vector') else emb2
        
        d = len(v1)
        dot_product = np.dot(v1, v2)
        # 将点乘结果映射为 0~1 的汉明相似度（1代表完全相同，0代表完全相反）
        return float((dot_product + d) / (2 * d))

    def preload_vectorized(self, samples: list[Identifier]) -> None:
        """
        Preloads all samples into one numpy ndarray for vectorized comparison.
        """
        # 提取向量特征进行堆叠
        vectors = []
        for s in samples:
            emb = self._embeddings.get(s)
            v = emb.vector if hasattr(emb, 'vector') else emb
            vectors.append(v)
            
        self._matrix = np.stack(vectors)

    def vectorized_similarity(self, sample: Identifier) -> np.ndarray:
        """
        Similarities for all the items in the preloaded vector.
        """
        emb = self._embeddings.get(sample)
        v = emb.vector if hasattr(emb, 'vector') else emb
        
        # 执行矩阵乘法（批量点乘）
        vals = np.matmul(self._matrix, v)
        
        # --- 修改点 2：移除截断，应用向量化汉明映射 ---
        d = self._matrix.shape[1]  # 向量的维度 D
        
        # 移除原来的 vals[vals < 0] = 0
        # 转化为规范化的相似度得分
        sim_scores = (vals + d) / (2 * d)
        
        return sim_scores