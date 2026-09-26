"""Multiple-choice quiz: quiz_muon_k2."""

TASK = {
    "type": "choice",
    "title": 'MuonClip Recipe',
    "title_zh": 'MuonClip 配方',
    "difficulty": 'Medium',
    "question_en": "Kimi K2's pretraining optimizer recipe is:",
    "question_zh": 'Kimi K2 的预训练优化器配方是：',
    "options": ['AdamW with cosine decay and loss-spike rollback', 'Muon plus QK-clip (clipping the QK root-mean-square outlier)', 'Lion with aggressive weight decay', 'Adafactor with ZeRO-3'],
    "answer": 1,
    "explanation_en": 'MuonClip = Muon + QK-clip; K2 trained 15.5T tokens with zero loss spikes — the first open recipe at that scale to run Muon for the bulk of pretraining.',
    "explanation_zh": 'MuonClip = Muon + QK-clip（裁剪 QK 的均方根离群值）；K2 用它训了 15.5T token、零 loss spike——首个在该规模上用 Muon 扛主体的开源配方。',
}
