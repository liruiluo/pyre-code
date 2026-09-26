"""Multiple-choice quiz: quiz_reverse_kl."""

TASK = {
    "type": "choice",
    "title": 'Reverse KL Shape',
    "title_zh": '反向 KL 的形态',
    "difficulty": 'Medium',
    "question_en": 'Compared with forward KL, reverse KL D_KL(student || teacher) is:',
    "question_zh": '与前向 KL 相比，反向 KL D_KL(student ‖ teacher) 的特点是：',
    "options": ['mass-covering — the student spreads over all teacher modes', 'mode-seeking — the student concentrates where the teacher is confident and may drop modes', 'symmetric to forward KL up to temperature', 'always larger in magnitude'],
    "answer": 1,
    "explanation_en": "Reverse KL explodes where the teacher has near-zero mass, so the student avoids those regions and concentrates on the teacher's dominant modes — the asymmetry on-policy distillation exploits.",
    "explanation_zh": '反向 KL 在教师概率近零处会爆炸，学生会避开这些区域、集中在教师的主模式上——在线策略蒸馏正是利用了这个不对称性。',
}
