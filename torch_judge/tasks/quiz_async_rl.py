"""Multiple-choice quiz: quiz_async_rl."""

TASK = {
    "type": "choice",
    "title": 'Async RL Decoupling',
    "title_zh": '异步 RL 解耦什么',
    "difficulty": 'Medium',
    "question_en": 'Async RL training systems (APPO / AReaL-style) primarily decouple:',
    "question_zh": '异步 RL 训练系统（APPO / AReaL 风格）主要解耦的是：',
    "options": ['the reward model from the policy', 'rollout generation from policy updates — workers keep sampling while the trainer steps, with importance ratios correcting off-policyness', 'CPU inference from GPU inference', 'training from evaluation checkpoints'],
    "answer": 1,
    "explanation_en": 'Rollout actors and the trainer run as loosely-coupled loops; weight sync lags behind, and PPO-style clipping with replay windows bounds the resulting staleness.',
    "explanation_zh": '采样 actor 和训练器是松耦合的两个循环；权重同步有滞后，用 PPO 式裁剪加回放窗口把陈旧度压在界内。',
}
