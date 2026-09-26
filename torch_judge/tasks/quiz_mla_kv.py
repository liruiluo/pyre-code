"""Multiple-choice quiz: quiz_mla_kv."""

TASK = {
    "type": "choice",
    "title": 'MLA KV Cache',
    "title_zh": 'MLA 的 KV 缓存',
    "difficulty": 'Medium',
    "question_en": 'MLA shrinks the KV cache by:',
    "question_zh": 'MLA 压缩 KV 缓存的方式是：',
    "options": ['quantizing keys and values to 4-bit', 'caching a compressed joint latent per token and re-deriving K/V from it at attention time', 'sharing one KV entry across all layers', 'dropping KV for sliding-window layers'],
    "answer": 1,
    "explanation_en": 'A low-rank joint projection compresses per-head K/V into one latent vector (plus the RoPE-decoupled part); only that is cached, and heads are re-derived when needed.',
    "explanation_zh": '低秩联合投影把逐头的 K/V 压成一个潜在向量（外加与 RoPE 解耦的部分）；只缓存它，需要时再还原各头。',
}
