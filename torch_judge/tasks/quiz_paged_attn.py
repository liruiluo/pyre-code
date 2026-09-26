"""Multiple-choice quiz: quiz_paged_attn."""

TASK = {
    "type": "choice",
    "title": 'PagedAttention Target',
    "title_zh": 'PagedAttention 治什么',
    "difficulty": 'Medium',
    "question_en": 'PagedAttention (vLLM) attacks which bottleneck?',
    "question_zh": 'PagedAttention（vLLM）解决的是哪个瓶颈？',
    "options": ['attention FLOPs', 'KV-cache fragmentation — cache blocks live in fixed-size pages like OS virtual memory', 'token embedding lookup latency', 'weight sharding across GPUs'],
    "answer": 1,
    "explanation_en": 'Fixed-size pages plus a block table eliminate internal and external fragmentation of the per-sequence KV cache, and enable copy-on-write prefix sharing.',
    "explanation_zh": '定长页 + 块表消除了逐序列 KV 缓存的内部与外部碎片，还顺带支持写时复制的前缀共享。',
}
