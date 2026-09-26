"""Multiple-choice quiz: quiz_zero_stages."""

TASK = {
    "type": "choice",
    "title": 'ZeRO Stage 3',
    "title_zh": 'ZeRO 第 3 级',
    "difficulty": 'Medium',
    "question_en": 'Compared with ZeRO stage 1, stage 3 additionally shards:',
    "question_zh": '相比 ZeRO 第 1 级，第 3 级额外切分了：',
    "options": ['activations', 'gradients and model parameters', 'optimizer states only', 'the dataset'],
    "answer": 1,
    "explanation_en": 'Stage 1 shards optimizer states; stage 2 adds gradients; stage 3 adds parameters — full sharding with per-layer all-gather in forward and backward.',
    "explanation_zh": '第 1 级切优化器状态；第 2 级加梯度；第 3 级加模型参数——全切分，前向/反向按层 all-gather。',
}
