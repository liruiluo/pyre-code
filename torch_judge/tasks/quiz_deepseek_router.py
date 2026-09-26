"""Multiple-choice quiz: quiz_deepseek_router."""

TASK = {
    "type": "choice",
    "title": 'Aux-Loss-Free Routing',
    "title_zh": '无辅助损失路由',
    "difficulty": 'Medium',
    "question_en": "DeepSeek-V3's aux-loss-free load balancing works by:",
    "question_zh": 'DeepSeek-V3 的无辅助损失（aux-loss-free）负载均衡靠的是：',
    "options": ['adding a KL term between the router distribution and uniform', 'adjusting a per-expert bias from expert load and adding it to routing scores for top-k selection only', 'randomly dropping overloaded experts each step', 'annealing the softmax temperature of the router'],
    "answer": 1,
    "explanation_en": 'The bias enters only the top-k selection, not the gate probability, so gradients stay clean; a tiny sequence-wise balance loss is kept only as a safeguard.',
    "explanation_zh": '偏置只进 top-k 选择、不进门控概率，因此不污染梯度；只保留一个很小的序列级平衡损失作为保险。',
}
