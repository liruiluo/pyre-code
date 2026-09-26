"""Multiple-choice quiz: quiz_dpo."""

TASK = {
    "type": "choice",
    "title": "DPO's Move",
    "title_zh": 'DPO 的做法',
    "difficulty": 'Easy',
    "question_en": "DPO's central move is:",
    "question_zh": 'DPO 的核心操作是：',
    "options": ['distilling preferences from a teacher model', 'optimizing preference pairs directly with a classification-style loss, skipping explicit reward-model training and RL', 'running PPO with a preference-shaped reward', 'weight-averaging preferred and rejected responses'],
    "answer": 1,
    "explanation_en": 'The Bradley-Terry likelihood re-derives into a loss over the log-ratio of chosen vs rejected policy probabilities; no reward model to sample, no rollout loop.',
    "explanation_zh": 'Bradley-Terry 似然重排成 chosen/rejected 策略概率对数比的损失；不用采样奖励模型，也不用 rollout 循环。',
}
