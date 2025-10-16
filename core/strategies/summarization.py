"""策略C: 基于摘要的压缩策略。"""

from __future__ import annotations

from typing import List, Tuple

from core.strategies.base import CompressionStrategy
from utils.text_cleaning import clean_document_text


class SummarizationStrategy(CompressionStrategy):
    """摘要压缩策略：将搜索结果压缩为累积更新的连贯文本摘要。"""

    def _init_knowledge_base(self) -> str:
        """初始化空字符串作为摘要。"""
        return ""

    def process(
        self,
        query: str,
        search_results: List[Tuple[str, str]],
    ) -> None:
        """清洗搜索结果，生成摘要并合并到现有摘要中。"""
        cleaned_contents = self._clean_search_results(search_results)
        new_summary = self._summarize_pages(query, cleaned_contents)

        if self.knowledge_base:
            self.knowledge_base = self._merge_summaries(
                self.knowledge_base, new_summary
            )
        else:
            self.knowledge_base = new_summary

    def get_checklist_context(self) -> str:
        """直接返回累积的摘要字符串。"""
        return self.knowledge_base

    # reconstruct_report_context 使用基类默认实现

    def _clean_search_results(
        self,
        search_results: List[Tuple[str, str]],
    ) -> List[Tuple[str, str]]:
        """清洗搜索结果内容。

        Args:
            search_results: (url, content) 元组列表

        Returns:
            (url, cleaned_content) 元组列表
        """
        raise NotImplementedError("需实现: 使用 clean_document_text 清洗所有 content")

    def _summarize_pages(
        self,
        query: str,
        cleaned_contents: List[Tuple[str, str]],
    ) -> str:
        """使用 LLM 生成当前批次搜索结果的摘要。

        Args:
            query: 查询文本
            cleaned_contents: (url, cleaned_content) 元组列表

        Returns:
            生成的摘要字符串
        """
        raise NotImplementedError("需实现: 调用 LLM 生成页面摘要")

    def _merge_summaries(self, old_summary: str, new_summary: str) -> str:
        """合并已有摘要和新摘要。

        Args:
            old_summary: 已有的累积摘要
            new_summary: 当前批次生成的新摘要

        Returns:
            合并后的摘要字符串
        """
        raise NotImplementedError("需实现: 合并两个摘要，避免重复信息")


__all__ = ["SummarizationStrategy"]
