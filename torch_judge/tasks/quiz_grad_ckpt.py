"""Multiple-choice quiz: quiz_grad_ckpt."""

TASK = {
    "type": "choice",
    "title": 'Gradient Checkpointing Trade',
    "title_zh": '梯度检查点的交易',
    "difficulty": 'Easy',
    "question_en": 'Gradient checkpointing trades:',
    "question_zh": '梯度检查点（gradient checkpointing）用的是什么交易？',
    "options": ['extra forward recomputation for lower activation memory', 'lower precision for faster matmuls', 'network bandwidth for disk space', 'stale gradients for throughput'],
    "answer": 0,
    "explanation_en": 'Store only layer inputs; recompute the forward activations during backward — roughly one extra forward pass buys a large cut in activation memory.',
    "explanation_zh": '只存每层输入，反向时重算前向激活——大约多一次前向的开销，换大幅下降的激活内存。',
}
