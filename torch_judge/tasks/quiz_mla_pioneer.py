"""Multiple-choice quiz: quiz_mla_pioneer."""

TASK = {
    "type": "choice",
    "title": 'MLA Origin',
    "title_zh": 'MLA 的出处',
    "difficulty": 'Easy',
    "question_en": 'Which open-weights family introduced Multi-head Latent Attention (MLA), compressing the KV cache into a small per-token latent?',
    "question_zh": '多头潜在注意力（MLA，把 KV 缓存压缩成逐 token 的低维潜在向量）是哪个开源模型家族首先提出的？',
    "options": ['Qwen', 'Llama', 'DeepSeek', 'Mistral'],
    "answer": 2,
    "explanation_en": 'DeepSeek-V2 (2024) introduced MLA: only a compressed joint latent is cached, and K/V heads are re-derived at attention time. DeepSeek-V3 keeps it, and Kimi K3 uses a gated variant.',
    "explanation_zh": 'DeepSeek-V2（2024）提出 MLA：只缓存压缩后的联合潜在向量，注意力计算时再还原 K/V 头。DeepSeek-V3 沿用，Kimi K3 用的是门控变体。',
}
