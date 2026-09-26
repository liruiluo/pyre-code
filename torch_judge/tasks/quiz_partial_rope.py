"""Multiple-choice quiz: quiz_partial_rope."""

TASK = {
    "type": "choice",
    "title": 'Partial RoPE',
    "title_zh": '部分 RoPE',
    "difficulty": 'Easy',
    "question_en": "In GLM-style 'partial RoPE', some attention layers:",
    "question_zh": 'GLM 式的 partial RoPE 指的是部分注意力层：',
    "options": ['rotate only the key vectors, leaving queries unrotated', "apply RoPE to only part of each head's dimension, leaving the rest position-free", 'use two different rotation frequencies per head', 'skip positional information in the value path only'],
    "answer": 1,
    "explanation_en": 'Partial RoPE rotates a slice of the head dimension and leaves the remainder unrotated, so one model can mix strongly-positional and position-free subspaces across layers.',
    "explanation_zh": 'Partial RoPE 只旋转头维度的前一段，剩下的不旋转（无位置信息），让同一模型在不同层混合强位置与位置无关的子空间。',
}
