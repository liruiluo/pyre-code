"""Multiple-choice quiz: quiz_kda_layers."""

TASK = {
    "type": "choice",
    "title": 'Kimi K3 Layer Mix',
    "title_zh": 'Kimi K3 的层混合',
    "difficulty": 'Hard',
    "question_en": 'Kimi K3 (2.8T total / 104B activated) mixes two attention families across its 93 layers. Which combination?',
    "question_zh": 'Kimi K3（总参 2.8T / 激活 104B）在 93 层里混用了两种注意力家族，是哪种组合？',
    "options": ['24 delta-rule (KDA) layers + 69 gated MLA layers', '69 delta-rule (KDA) layers + 24 gated MLA layers', 'All gated MLA layers + 1 dense layer', 'All sliding-window layers + full attention'],
    "answer": 1,
    "explanation_en": 'K3 runs 69 KDA (Kimi Delta Attention, a delta-rule linear attention) + 24 gated MLA + 1 dense layer = 93; the 1M context rides on this hybrid.',
    "explanation_zh": 'K3 = 69 层 KDA（Kimi Delta Attention，delta 规则线性注意力）+ 24 层门控 MLA + 1 层稠密 = 93 层；1M 上下文正是靠这套混合栈撑起来的。',
}
