"""Multiple-choice quiz: quiz_opsd_views."""

TASK = {
    "type": "choice",
    "title": "OPSD's Two Views",
    "title_zh": 'OPSD 的双视图',
    "difficulty": 'Medium',
    "question_en": 'On-Policy Self-Distillation (OPSD) makes one model both teacher and student by:',
    "question_zh": '在线策略自蒸馏（OPSD）让一个模型同时当教师和学生，其做法是：',
    "options": ['averaging the weights of two checkpoints each step', 'conditioning two views differently — a privileged view sees the question plus a reference, the deployable view sees the question only — and pulling the deployable view toward the privileged one on its own samples', 'training the model to predict its own next-layer features', 'fine-tuning on self-generated data filtered by a reward model'],
    "answer": 1,
    "explanation_en": "The hint-conditioned view provides the target distribution; a stop-gradient KL on the student's own rollouts transfers the capability, and at deployment the hint is simply gone.",
    "explanation_zh": '带提示的视图提供目标分布；在学生自己的 rollout 上做 stop-gradient KL 把能力迁移过去；部署时提示直接消失。',
}
