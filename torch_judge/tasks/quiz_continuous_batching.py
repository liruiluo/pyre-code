"""Multiple-choice quiz: quiz_continuous_batching."""

TASK = {
    "type": "choice",
    "title": 'Continuous Batching',
    "title_zh": '连续批处理',
    "difficulty": 'Easy',
    "question_en": 'Continuous batching raises inference throughput because:',
    "question_zh": '连续批处理（continuous batching）提升推理吞吐的原因是：',
    "options": ['it batches only equal-length sequences', "finished requests exit and new requests join at every iteration — no padding to the batch's longest sequence", 'it sorts requests by length once at admission', 'it prefixes all requests with a shared system prompt'],
    "answer": 1,
    "explanation_en": 'The scheduler swaps sequences at token-level granularity, keeping the GPU busy; static batches waste compute on padding and stranded slots.',
    "explanation_zh": '调度器以 token 级粒度换入换出序列，GPU 始终有活干；静态批处理把算力浪费在 padding 和空位上。',
}
