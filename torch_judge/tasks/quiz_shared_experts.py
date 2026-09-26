"""Multiple-choice quiz: quiz_shared_experts."""

TASK = {
    "type": "choice",
    "title": 'Shared Experts in 2026',
    "title_zh": '2026 旗舰的共享专家',
    "difficulty": 'Medium',
    "question_en": 'Among 2025-2026 open-weight flagships, which statement about shared experts is TRUE?',
    "question_zh": '关于 2025-2026 开源旗舰的共享专家（shared expert），哪句是对的？',
    "options": ['Qwen3-MoE layers carry one shared expert each', 'MiMo-V2.6 ships 2 shared experts per MoE layer', 'Kimi K3 ships no shared experts', 'MiMo-V2.6 ships none, while Kimi K3 keeps two'],
    "answer": 3,
    "explanation_en": 'MiMo-V2.6: 384 routed experts, top-8, zero shared. Qwen3-MoE also ships without shared experts; DeepSeek-V3 keeps 1; Kimi K3 keeps 2. Shared experts are a design axis, not a constant.',
    "explanation_zh": 'MiMo-V2.6：384 个路由专家、top-8、无共享专家。Qwen3-MoE 同样没有共享专家；DeepSeek-V3 保留 1 个；Kimi K3 保留 2 个。共享专家是一个设计轴，不是常数。',
}
