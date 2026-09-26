"""Multiple-choice quiz: quiz_grpo_baseline."""

TASK = {
    "type": "choice",
    "title": 'GRPO Baseline',
    "title_zh": 'GRPO 的基线',
    "difficulty": 'Easy',
    "question_en": 'GRPO removes the PPO value network by using:',
    "question_zh": 'GRPO 去掉 PPO 的价值网络，改用：',
    "options": ['a learned reward model as baseline', 'the mean return of a group of responses sampled for the same prompt', 'no baseline at all', 'the KL penalty as the baseline'],
    "answer": 1,
    "explanation_en": "Advantage = reward minus the group mean (optionally std-normalized); one prompt's multiple samples become their own baseline — cheaper than training a separate critic.",
    "explanation_zh": '优势 = 奖励减去组均值（可选做标准差归一化）；同一提示的一组样本互为基线——比训练独立 critic 便宜。',
}
