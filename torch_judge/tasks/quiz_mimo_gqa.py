"""Multiple-choice quiz: quiz_mimo_gqa."""

TASK = {
    "type": "choice",
    "title": 'MiMo GQA Ratio',
    "title_zh": 'MiMo 的 GQA 配比',
    "difficulty": 'Hard',
    "question_en": 'Xiaomi MiMo-V2.6-Pro pairs 128 query heads with how many KV heads?',
    "question_zh": '小米 MiMo-V2.6-Pro 用 128 个 query 头配了多少个 KV 头？',
    "options": ['128', '32', '8', '1 (MQA)'],
    "answer": 2,
    "explanation_en": '128 query heads to 8 KV heads — an aggressive GQA ratio; the stack also runs 60 sliding-window layers to 10 global ones.',
    "explanation_zh": '128 个 query 头对 8 个 KV 头——非常激进的 GQA 配比；整栈还有 60 层滑窗注意力对 10 层全局注意力。',
}
