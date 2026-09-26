"""Multiple-choice quiz: quiz_qwen_vocab."""

TASK = {
    "type": "choice",
    "title": 'Qwen3.5/3.8 Vocab',
    "title_zh": 'Qwen3.5/3.8 词表',
    "difficulty": 'Hard',
    "question_en": 'The 2026-generation Qwen3.5/Qwen3.8 models enlarged the tokenizer vocabulary to roughly:',
    "question_zh": '2026 代的 Qwen3.5/Qwen3.8 把分词器词表扩大到了大约：',
    "options": ['~130K', '~152K', '~248K', '~400K'],
    "answer": 2,
    "explanation_en": 'Qwen3.5-4B and Qwen3.8-27B share a 248,320-entry vocabulary, up from ~152K in the Qwen2/3 era — shorter sequences for CJK and code.',
    "explanation_zh": 'Qwen3.5-4B 与 Qwen3.8-27B 共用 248,320 的词表，比 Qwen2/3 时代的 ~152K 大幅扩容——中文与代码的序列更短。',
}
