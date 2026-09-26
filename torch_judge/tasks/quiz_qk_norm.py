"""Multiple-choice quiz: quiz_qk_norm."""

TASK = {
    "type": "choice",
    "title": 'QK-Norm Purpose',
    "title_zh": 'QK-Norm 的作用',
    "difficulty": 'Easy',
    "question_en": 'QK-Norm (as used in Qwen3 and Gemma-3) exists to:',
    "question_zh": 'QK-Norm（Qwen3、Gemma-3 都在用）的目的是：',
    "options": ['normalize the attention output distribution', 'stabilize attention logits by L2-normalizing queries and keys before the dot product', 'replace RoPE with a learned positional bias', 'reduce the KV cache size'],
    "answer": 1,
    "explanation_en": 'L2-normalizing Q and K bounds the logit magnitude, preventing attention-logit explosion in deep models — a cheap, parameter-free stabilizer.',
    "explanation_zh": '对 Q、K 做 L2 归一化可以限制注意力分数的大小，防止深层模型的 logits 爆炸——一个零参数的训练稳定器。',
}
