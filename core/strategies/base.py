"""压缩策略的抽象基类。"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from core.types import KnowledgeBase
from utils.chunking import chunk_documents


class CompressionStrategy(ABC):
    """上下文压缩策略的抽象基类，定义所有压缩策略的统一接口。"""

    def __init__(self) -> None:
        """初始化策略实例，创建空的 chunk_store 和 knowledge_base。"""
        self.chunk_store: Dict[str, str] = {}
        self.chunk_url_map: Dict[str, str] = {}  # chunk_id -> url 映射
        self.knowledge_base: KnowledgeBase = self._init_knowledge_base()

    @abstractmethod
    def _init_knowledge_base(self) -> KnowledgeBase:
        """初始化空知识库，返回策略专属的初始数据结构。"""
        raise NotImplementedError

    @abstractmethod
    def process(
        self,
        query: str,
        search_results: List[Tuple[str, str]],
    ) -> None:
        """处理搜索结果并更新知识库。

        Args:
            query: 查询文本
            search_results: (url, content) 元组列表
        """
        raise NotImplementedError

    @abstractmethod
    def get_checklist_context(self) -> str | List[Dict]:
        """返回供 Checklist Agent 使用的压缩上下文（字符串或字典列表）。"""
        raise NotImplementedError

    def reconstruct_report_context(
        self,
        relevant_items: List[str] | None = None,
    ) -> str:
        """重建供报告生成器使用的完整上下文。

        默认实现直接返回 get_checklist_context() 的结果。
        策略A需要重写以根据 relevant_items 追溯原始 chunks。

        Args:
            relevant_items: 可选的相关项标识符列表

        Returns:
            完整的报告上下文字符串
        """
        result = self.get_checklist_context()
        return result if isinstance(result, str) else str(result)

    def _generate_chunk_id(self, url: str, index: int) -> str:
        """生成唯一的 chunk ID，格式为 {url_hash}-{index}。"""
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        return f"{url_hash}-{index}"

    def _chunk_and_store(
        self,
        search_results: List[Tuple[str, str]],
    ) -> List[Tuple[str, str]]:
        """对搜索结果进行分块、生成ID并存储到 chunk_store。

        Args:
            search_results: (url, cleaned_content) 元组列表，内容应已清洗

        Returns:
            (chunk_id, chunk_text) 元组列表
        """

        chunk_pairs: List[Tuple[str, str]] = []

        for url, content in search_results:
            if not content or not content.strip():
                continue

            # 分块
            chunks = chunk_documents(content)

            # 为每个chunk生成ID并存储
            for index, chunk_text in enumerate(chunks):
                chunk_id = self._generate_chunk_id(url, index)
                self.chunk_store[chunk_id] = chunk_text
                self.chunk_url_map[chunk_id] = url  # 存储 chunk_id 到 url 的映射
                chunk_pairs.append((chunk_id, chunk_text))

        return chunk_pairs


__all__ = ["CompressionStrategy"]
