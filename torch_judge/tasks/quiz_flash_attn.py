"""Multiple-choice quiz: quiz_flash_attn."""

TASK = {
    "type": "choice",
    "title": 'FlashAttention Gain',
    "title_zh": 'FlashAttention 的收益来源',
    "difficulty": 'Easy',
    "question_en": "FlashAttention's speedup over naive attention mainly comes from:",
    "question_zh": 'FlashAttention 比朴素注意力快，主要因为：',
    "options": ['approximating softmax with a polynomial', 'IO-awareness — tiling the computation so the full attention matrix never materializes in HBM', 'quantizing keys and values to int8', 'skipping causal-masked upper-triangle FLOPs only'],
    "answer": 1,
    "explanation_en": 'Same FLOPs, vastly fewer HBM reads/writes: tiles stay in SRAM with an online-softmax rescale. It computes exact attention — memory traffic was the real bottleneck.',
    "explanation_zh": 'FLOPs 不变、HBM 读写大减：分块驻留 SRAM，在线 softmax 逐步缩放。算的是精确注意力——瓶颈本来就在访存。',
}
