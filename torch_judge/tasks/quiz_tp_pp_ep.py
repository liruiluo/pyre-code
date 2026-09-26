"""Multiple-choice quiz: quiz_tp_pp_ep."""

TASK = {
    "type": "choice",
    "title": 'Expert Parallelism',
    "title_zh": '专家并行',
    "difficulty": 'Easy',
    "question_en": 'Placing different MoE experts on different devices is called:',
    "question_zh": '把不同的 MoE 专家放在不同设备上叫：',
    "options": ['Tensor parallelism', 'Pipeline parallelism', 'Expert parallelism', 'Sequence parallelism'],
    "answer": 2,
    "explanation_en": 'TP splits each matmul across devices; PP splits layers into stages; EP shards experts — all-to-all dispatches route tokens to the devices owning each expert.',
    "explanation_zh": 'TP 按设备切分单个矩阵乘；PP 把层切成流水段；EP 切分专家——用 all-to-all 把 token 送到专家所在的设备。',
}
