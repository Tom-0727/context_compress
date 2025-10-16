"""策略B: 基于块过滤的压缩策略。"""

import time
import json
from pathlib import Path
from typing import List, Tuple

from core.strategies.base import CompressionStrategy
from utils.llm_client import call_llm


class ChunkFilteringStrategy(CompressionStrategy):
    """块过滤压缩策略：通过 LLM 判断并保留与查询相关的文本块。"""

    def __init__(self, verbose: bool = False) -> None:
        """初始化策略实例。

        Args:
            verbose: 是否打印耗时信息
        """
        super().__init__()
        self.verbose = verbose

    def _init_knowledge_base(self) -> List[str]:
        """初始化空的 chunk_id 列表。"""
        return []

    def process(
        self,
        query: str,
        search_results: List[Tuple[str, str]],
    ) -> None:
        """对搜索结果进行 chunking，过滤相关块并累积到知识库。"""

        start_time = time.time()
        chunk_pairs = self._chunk_and_store(search_results)
        relevant_chunk_ids = self._filter_chunks(query, chunk_pairs)
        self.knowledge_base.extend(relevant_chunk_ids)

        if self.verbose:
            elapsed = time.time() - start_time
            print(f"⏱️  process() 耗时: {elapsed:.2f}s")

    def get_checklist_context(self) -> str:
        """从 chunk_store 提取所有相关 chunks 的原文并拼接。"""
        if not self.knowledge_base:
            return ""

        chunks = []
        for chunk_id in self.knowledge_base:
            if chunk_id in self.chunk_store:
                chunks.append(self.chunk_store[chunk_id])

        return "\n\n---\n\n".join(chunks)

    # reconstruct_report_context 使用基类默认实现

    def _filter_chunks(
        self,
        query: str,
        chunk_pairs: List[Tuple[str, str]],
    ) -> List[str]:
        """使用 LLM 判断哪些 chunks 与 query 相关。

        Args:
            query: 查询文本
            chunk_pairs: (chunk_id, chunk_text) 元组列表

        Returns:
            相关的 chunk_id 列表
        """
        

        if not chunk_pairs:
            return []

        # 读取prompt模板
        prompt_file = Path(__file__).parent.parent.parent / "prompts" / "chunk_filter.txt"
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt_template = f.read()

        # 构造chunks JSON（使用索引而非 chunk_id）
        chunks_data = [
            {"index": i, "text": text} for i, (chunk_id, text) in enumerate(chunk_pairs)
        ]
        chunks_json = json.dumps(chunks_data, ensure_ascii=False, indent=2)

        # 填充prompt
        prompt = prompt_template.format(query=query, chunks=chunks_json)

        # 调用LLM
        messages = [{"role": "user", "content": prompt}]
        response = call_llm(
            messages=messages,
            response_format={"type": "json_object"},
        )

        # 解析响应，将索引映射回 chunk_id
        result = json.loads(response)
        relevant_indices = result.get("relevant_indices", [])
        return [chunk_pairs[i][0] for i in relevant_indices if i < len(chunk_pairs)]


__all__ = ["ChunkFilteringStrategy"]
