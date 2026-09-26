"""Multiple-choice quiz: quiz_mtp_drafter."""

TASK = {
    "type": "choice",
    "title": 'MiMo MTP Drafter',
    "title_zh": 'MiMo 的 MTP 起草器',
    "difficulty": 'Medium',
    "question_en": "MiMo-V2.6's MTP drafter — a 5-layer sliding-window model — proposes how many tokens per verification step?",
    "question_zh": 'MiMo-V2.6 的 MTP 起草器（一个 5 层滑窗模型）每步验证提议几个 token？',
    "options": ['2', '3', '5', '7'],
    "answer": 3,
    "explanation_en": 'The drafter proposes 7 tokens per step; accepted prefixes advance the sequence in a single round, amortizing decode cost.',
    "explanation_zh": '起草器每步提议 7 个 token；被接受的前缀让序列一轮前进多步，摊薄解码开销。',
}
