"""Multiple-choice quiz: quiz_opd_property."""

TASK = {
    "type": "choice",
    "title": 'On-Policy Distillation Core',
    "title_zh": '在线策略蒸馏核心',
    "difficulty": 'Easy',
    "question_en": 'The defining property of on-policy distillation is:',
    "question_zh": '在线策略蒸馏（on-policy distillation）的定义性特征是：',
    "options": ['the student trains on sequences sampled from its own current policy, with the teacher scoring those tokens', 'the teacher generates the training sequences for the student', 'the student imitates a dataset distilled once from the teacher', 'the loss is computed only on tool-call tokens'],
    "answer": 0,
    "explanation_en": "On-policy = the rollout distribution is the student's own; the teacher only scores, giving a dense per-token signal that behaves like RL with a learned reward — far cheaper than reward-model sampling.",
    "explanation_zh": 'On-policy 指 rollout 分布来自学生自身；教师只打分，给出逐 token 的稠密信号——效果像带学习奖励的 RL，却比采样奖励模型便宜得多。',
}
