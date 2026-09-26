"""Multiple-choice quiz: quiz_mopd."""

TASK = {
    "type": "choice",
    "title": 'MOPD Meaning',
    "title_zh": 'MOPD 指什么',
    "difficulty": 'Medium',
    "question_en": "In Xiaomi MiMo-V2.6's post-training, MOPD stands for:",
    "question_zh": '小米 MiMo-V2.6 后训练里的 MOPD 指的是：',
    "options": ['Mixed-Precision Policy Distillation', 'Multi-Prefix Multi-Teacher On-Policy Distillation', 'Masked Output Projection Distillation', 'Model-Optimized Preference Distillation'],
    "answer": 1,
    "explanation_en": "MOPD = Multi-Prefix Multi-Teacher On-Policy Distillation — capability-mixing weights blend several teachers' scores on the student's own rollouts. The exact weighting formula is not public.",
    "explanation_zh": 'MOPD = Multi-Prefix Multi-Teacher On-Policy Distillation（多前缀多教师在线策略蒸馏）——用能力组合权重融合多个教师在学生自己 rollout 上的打分；精确加权公式未公开。',
}
