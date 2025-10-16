"""策略C: 基于摘要的压缩策略。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

from core.strategies.base import CompressionStrategy
from utils.llm_client import call_llm


class SummarizationStrategy(CompressionStrategy):
    """摘要压缩策略：将搜索结果压缩为累积更新的连贯文本摘要。"""

    def __init__(self, verbose: bool = False) -> None:
        """初始化策略实例。

        Args:
            verbose: 是否打印耗时信息
        """
        super().__init__()
        self.verbose = verbose

    def _init_knowledge_base(self) -> str:
        """初始化空字符串作为摘要。"""
        return ""

    def process(
        self,
        query: str,
        search_results: List[Tuple[str, str]],
    ) -> None:
        """生成摘要并累积更新到 knowledge_base。

        Args:
            query: 查询文本
            search_results: (url, cleaned_content) 元组列表（已清洗）
        """
        import time

        start_time = time.time()

        # 生成当前批次的摘要
        new_summary = self._summarize_pages(query, search_results)

        # 合并到已有摘要
        if self.knowledge_base:
            self.knowledge_base = self._merge_summaries(
                self.knowledge_base, new_summary
            )
        else:
            self.knowledge_base = new_summary

        if self.verbose:
            elapsed = time.time() - start_time
            print(f"⏱️  process() 耗时: {elapsed:.2f}s")

    def get_checklist_context(self) -> str:
        """直接返回累积的摘要字符串。"""
        return self.knowledge_base

    # reconstruct_report_context 使用基类默认实现

    def _summarize_pages(
        self,
        query: str,
        search_results: List[Tuple[str, str]],
    ) -> str:
        """使用 LLM 生成当前批次搜索结果的摘要。

        Args:
            query: 查询文本
            search_results: (url, cleaned_content) 元组列表

        Returns:
            生成的摘要字符串
        """
        if not search_results:
            return ""

        # 读取 prompt 模板
        prompt_file = Path(__file__).parent.parent.parent / "prompts" / "page_summarize.txt"
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt_template = f.read()

        # 构造页面内容 JSON（限制每页长度避免超token）
        pages_data = [
            {"url": url, "content": content[:5000]}
            for url, content in search_results
        ]
        pages_json = json.dumps(pages_data, ensure_ascii=False, indent=2)

        # 填充 prompt
        prompt = prompt_template.format(query=query, pages=pages_json)

        # 调用 LLM（不需要 json_object 格式，直接返回文本）
        messages = [{"role": "user", "content": prompt}]
        response = call_llm(messages=messages)

        return response.strip()

    def _merge_summaries(self, old_summary: str, new_summary: str) -> str:
        """合并已有摘要和新摘要。

        Args:
            old_summary: 已有的累积摘要
            new_summary: 当前批次生成的新摘要

        Returns:
            合并后的摘要字符串
        """
        if not old_summary:
            return new_summary
        if not new_summary:
            return old_summary

        # 读取合并 prompt 模板
        prompt_file = Path(__file__).parent.parent.parent / "prompts" / "summary_merge.txt"
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt_template = f.read()

        # 填充 prompt
        prompt = prompt_template.format(
            old_summary=old_summary,
            new_summary=new_summary
        )

        # 调用 LLM
        messages = [{"role": "user", "content": prompt}]
        response = call_llm(messages=messages)

        return response.strip()


__all__ = ["SummarizationStrategy"]
