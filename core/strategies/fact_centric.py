"""策略A: 基于事实提取的压缩策略。"""

from __future__ import annotations

from typing import List, Tuple

from core.strategies.base import CompressionStrategy
from core.types import FactObject


class FactCentricStrategy(CompressionStrategy):
    """事实中心压缩策略：从文本块中提取结构化事实并保持原文追溯能力。"""

    def _init_knowledge_base(self) -> List[FactObject]:
        """初始化空的事实列表。"""
        return []

    def process(
        self,
        query: str,
        search_results: List[Tuple[str, str]],
    ) -> None:
        """对搜索结果进行 chunking，提取事实并累积到知识库。"""
        chunk_pairs = self._chunk_and_store(search_results)
        new_facts = self._extract_facts(query, chunk_pairs)
        self.knowledge_base.extend(new_facts)

    def get_checklist_context(self) -> List[dict]:
        """返回事实对象的字典列表。"""
        raise NotImplementedError("需实现: 将 FactObject 转换为字典列表")

    def reconstruct_report_context(
        self,
        relevant_items: List[str] | None = None,
    ) -> str:
        """根据选中的事实 IDs 从 chunk_store 重建完整上下文。"""
        raise NotImplementedError("需实现: 根据 relevant_items 从 chunk_store 重建上下文")

    def _extract_facts(
        self,
        query: str,
        chunk_pairs: List[Tuple[str, str]],
    ) -> List[FactObject]:
        """从 chunks 中提取与 query 相关的事实。

        Args:
            query: 查询文本
            chunk_pairs: (chunk_id, chunk_text) 元组列表

        Returns:
            提取的 FactObject 列表
        """
        raise NotImplementedError("需实现: 调用 LLM 进行事实提取")


__all__ = ["FactCentricStrategy"]
