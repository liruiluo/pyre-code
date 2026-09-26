"""Multiple-choice quiz: quiz_1m_context."""

TASK = {
    "type": "choice",
    "title": '1M Context Club',
    "title_zh": '1M 上下文俱乐部',
    "difficulty": 'Medium',
    "question_en": 'Which pair of 2026 open-weight flagships advertise ~1M-token context windows?',
    "question_zh": '哪一对 2026 年开源旗舰宣称支持约 1M token 的上下文窗口？',
    "options": ['Kimi K3 and GLM-5.3', 'MiMo-V2.6 and Qwen3.8', 'DeepSeek-V3 and Llama-4', 'None — no open model exceeds 256K'],
    "answer": 0,
    "explanation_en": 'Kimi K3 (hybrid KDA stack) and GLM-5.3 (753B, DSA-based MoE) both claim 1M-token context in 2026.',
    "explanation_zh": 'Kimi K3（KDA 混合栈）和 GLM-5.3（753B，DSA 稀疏注意力 MoE）在 2026 年都宣称 1M token 上下文。',
}
