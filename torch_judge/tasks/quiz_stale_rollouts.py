"""Multiple-choice quiz: quiz_stale_rollouts."""

TASK = {
    "type": "choice",
    "title": 'Stale Sample Correction',
    "title_zh": '陈旧样本的修正',
    "difficulty": 'Hard',
    "question_en": 'In async RL, rollouts sampled under older policy versions are typically corrected by:',
    "question_zh": '异步 RL 里，用旧版策略采出来的 rollout 通常靠什么修正？',
    "options": ['discarding any sample older than one update', 'importance-sampling ratios between the current and sampling policies, with clipping bounding the correction', 'restarting the rollout workers on every update', 'doubling the learning rate on fresh samples'],
    "answer": 1,
    "explanation_en": 'The ratio of current over sampling policy reweights old samples; PPO-style clipping caps the per-token correction — and the ratio magnitude itself is the staleness signal.',
    "explanation_zh": '用当前策略与采样策略的概率比给旧样本重加权；PPO 式裁剪封顶逐 token 修正量——比值的大小本身就是陈旧度信号。',
}
