"""Multiple-choice quiz: quiz_prefill_decode."""

TASK = {
    "type": "choice",
    "title": 'Prefill/Decode Split',
    "title_zh": '预填充/解码分离',
    "difficulty": 'Medium',
    "question_en": 'Disaggregating prefill and decode into separate serving pools pays off because:',
    "question_zh": '把预填充（prefill）和解码（decode）拆到不同服务池划算的原因是：',
    "options": ['prefill is compute-bound while decode is memory-bandwidth-bound', 'prefill requires lower precision than decode', 'decode cannot run on the same GPUs as prefill', 'prefill outputs tokens while decode processes prompts'],
    "answer": 0,
    "explanation_en": 'Prefill saturates FLOPs with large matmuls over the prompt; decode emits one token at a time and is bound by KV-cache reads — different bottlenecks want different pool sizes and hardware.',
    "explanation_zh": '预填充是算力型（整段提示的大矩阵乘）；解码一次吐一个 token、受 KV 缓存读取带宽限制——瓶颈不同，池子的规模和硬件配比也不同。',
}
