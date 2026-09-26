"""Agent memory bank scoring task."""

TASK = {
    "title": "Score an Agent Memory Bank",
    "title_zh": "给 Agent 记忆库打分",
    "difficulty": "Medium",
    "description_en": "Implement the retrieval scorer of an agent memory bank: relevance times recency.\n\nLong-running agents keep a bank of past notes, each stored with a timestamp and a content vector. At recall time, each note gets `score = cosine_similarity(query, note) * exp(-decay_rate * (now - timestamp))` — relevant but stale notes fade out. Return the indices of the top-k notes, best first. This is the retrieval core of MemGPT-style memory systems and every agent memory middleware.\n\n**Signature:** `score_memory(query, memories, timestamps, current_time, decay_rate, topk) -> list[int]`\n\n**Parameters:**\n- `query` — content vector, shape `(d,)`\n- `memories` — bank of note vectors, shape `(N, d)`\n- `timestamps` — write time of each note, shape `(N,)`\n- `current_time`, `decay_rate` — scalars\n- `topk` — number of notes to return\n\n**Returns:** up to `topk` indices, sorted by score descending. Ties break by lower index (older entry wins).\n\n**Constraints:**\n- `topk <= 0` or an empty bank returns `[]`\n- `topk > N` returns all `N` notes, still sorted by score\n- Cosine similarity uses the L2-normalized vectors; zero vectors score 0",
    "description_zh": "实现 agent 记忆库的检索打分：相关性乘以新鲜度。\n\n长程 agent 把历史笔记存成记忆库，每条带时间戳和内容向量。召回时每条笔记得分 `score = cosine_similarity(query, note) * exp(-decay_rate * (now - timestamp))`——相关但陈旧的笔记会淡出。返回 top-k 笔记的索引，分数最高的在前。这是 MemGPT 式记忆系统和各类 agent 记忆中间件的检索核心。\n\n**签名:** `score_memory(query, memories, timestamps, current_time, decay_rate, topk) -> list[int]`\n\n**参数:**\n- `query` — 内容向量，形状 `(d,)`\n- `memories` — 笔记向量库，形状 `(N, d)`\n- `timestamps` — 每条笔记的写入时间，形状 `(N,)`\n- `current_time`、`decay_rate` — 标量\n- `topk` — 返回条数\n\n**返回:** 至多 `topk` 个索引，按分数降序。平分时取较小索引（更早的条目优先）。\n\n**约束:**\n- `topk <= 0` 或空库返回 `[]`\n- `topk > N` 返回全部 `N` 条，仍按分数排序\n- 余弦相似度用 L2 归一化向量；零向量得 0 分",
    "function_name": "score_memory",
    "hint": "`F.cosine_similarity(query.unsqueeze(0), memories)` gives all relevances; multiply by `torch.exp(-decay_rate * (current_time - timestamps))`; sort with `torch.argsort(scores, descending=True, stable=True)`.",
    "hint_zh": "`F.cosine_similarity(query.unsqueeze(0), memories)` 一次算出全部相关性；乘 `torch.exp(-decay_rate * (current_time - timestamps))`；用 `torch.argsort(scores, descending=True, stable=True)` 排序。",
    "tests": [
        {
            "name": "Hand-computed scores and ordering",
            "code": """
import torch, math
query = torch.tensor([1.0, 0.0])
memories = torch.tensor([
    [1.0, 0.0],   # cos = 1.0
    [0.0, 1.0],   # cos = 0.0
    [1.0, 1.0],   # cos = 1/sqrt(2)
])
timestamps = torch.tensor([0.0, 0.0, 0.0])
now, decay = 0.0, 0.5
out = {fn}(query, memories, timestamps, now, decay, 3)
assert out[0] == 0, f'Most relevant note should rank first, got {out}'
assert out[1] == 2, f'Second should be the 45-degree note, got {out}'
assert out[2] == 1, f'Orthogonal note ranks last, got {out}'
""",
        },
        {
            "name": "Decay downranks stale but identical content",
            "code": """
import torch
query = torch.tensor([1.0, 1.0])
memories = torch.tensor([[2.0, 2.0], [2.0, 2.0]])
timestamps = torch.tensor([0.0, 10.0])
out = {fn}(query, memories, timestamps, current_time=10.0, decay_rate=0.1, topk=2)
assert out == [1, 0], f'Fresh copy (idx 1) must outrank the stale one, got {out}'
""",
        },
        {
            "name": "Top-k truncation",
            "code": """
import torch
query = torch.tensor([1.0, 0.0])
memories = torch.eye(2)
timestamps = torch.zeros(2)
out = {fn}(query, memories, timestamps, 0.0, 0.1, 1)
assert out == [0], f'Expected only the best note, got {out}'
""",
        },
        {
            "name": "topk larger than bank returns all sorted",
            "code": """
import torch, math
query = torch.tensor([0.0, 1.0])
memories = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
timestamps = torch.tensor([5.0, 5.0, 5.0])
out = {fn}(query, memories, timestamps, current_time=5.0, decay_rate=1.0, topk=10)
assert out[0] == 1, f'Aligned note first, got {out}'
assert sorted(out) == [0, 1, 2], f'All notes must be returned, got {out}'
""",
        },
        {
            "name": "Empty bank and non-positive topk",
            "code": """
import torch
query = torch.tensor([1.0, 0.0])
assert {fn}(query, torch.zeros(0, 2), torch.zeros(0), 0.0, 0.1, 5) == []
assert {fn}(query, torch.eye(2), torch.zeros(2), 0.0, 0.1, 0) == []
assert {fn}(query, torch.eye(2), torch.zeros(2), 0.0, 0.1, -1) == []
""",
        },
        {
            "name": "Zero memory vector scores zero and does not NaN",
            "code": """
import torch
query = torch.tensor([1.0, 0.0])
memories = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
timestamps = torch.zeros(2)
out = {fn}(query, memories, timestamps, 0.0, 0.1, 2)
assert out == [1, 0], f'Zero vector must not crash and must rank last, got {out}'
""",
        },
        {
            "name": "Matches the reference formula exactly",
            "code": """
import torch
torch.manual_seed(0)
query = torch.randn(8)
memories = torch.randn(5, 8)
timestamps = torch.rand(5) * 100
now, decay = 100.0, 0.02
scores = torch.nn.functional.cosine_similarity(query.unsqueeze(0), memories) * torch.exp(-decay * (now - timestamps))
expected = torch.argsort(scores, descending=True, stable=True).tolist()
out = {fn}(query, memories, timestamps, now, decay, 5)
assert out == expected, f'Got {out}, expected {expected}'
""",
        },
    ],
    "solution": '''import torch
import torch.nn.functional as F


def score_memory(query, memories, timestamps, current_time, decay_rate, topk):
    if topk is None or topk <= 0 or memories.numel() == 0 or memories.shape[0] == 0:
        return []
    relevance = F.cosine_similarity(query.unsqueeze(0), memories, dim=1)
    recency = torch.exp(-decay_rate * (current_time - timestamps))
    scores = relevance * recency
    order = torch.argsort(scores, descending=True, stable=True)
    k = min(int(topk), memories.shape[0])
    return order[:k].tolist()''',
    "demo": """import torch
query = torch.tensor([1.0, 0.0])
memories = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]])
timestamps = torch.tensor([0.0, 1.0, 9.0])
print(score_memory(query, memories, timestamps, current_time=10.0, decay_rate=0.2, topk=2))""",
}
