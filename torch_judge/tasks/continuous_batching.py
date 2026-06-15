"""Continuous batching scheduler task."""

TASK = {
    "title": "Continuous Batching Schedule",
    "title_zh": "连续批处理调度",
    "difficulty": "Medium",
    "description_en": "Implement a small continuous-batching scheduler for autoregressive serving.\n\nUnlike static batching, continuous batching admits new requests at every decoding step as soon as capacity is available.\n\n**Signature:** `continuous_batching_schedule(requests, max_batch_size) -> list[list[int]]`\n\n**Parameters:**\n- `requests` — list of `(arrival_step, num_decode_tokens)` tuples\n- `max_batch_size` — maximum active requests per step\n\n**Returns:** a list where each item is the sorted request ids active during that decoding step\n\n**Constraints:**\n- Request ids are their original list indices\n- At each step, admit arrived waiting requests in FIFO/id order until capacity is full\n- Each active request consumes one decode token per step\n- Remove requests whose remaining tokens reach zero at the end of the step\n- Continue until all requests have finished",
    "description_zh": "实现一个自回归推理服务中的连续批处理调度器。\n\n不同于静态 batching，连续批处理会在每个解码步只要有容量就接入新请求。\n\n**签名:** `continuous_batching_schedule(requests, max_batch_size) -> list[list[int]]`\n\n**参数:**\n- `requests` — `(arrival_step, num_decode_tokens)` 元组列表\n- `max_batch_size` — 每步最多同时活跃请求数\n\n**返回:** 列表中每一项是该解码步活跃的请求 id，按升序排列\n\n**约束:**\n- 请求 id 为其在原始列表中的下标\n- 每个 step 按 FIFO/id 顺序接入已到达等待请求，直到容量满\n- 每个活跃请求每步消耗一个 decode token\n- 剩余 token 到 0 的请求在该步结束时移除\n- 一直运行到所有请求完成",
    "function_name": "continuous_batching_schedule",
    "hint": "Track `time`, `waiting`, `active`, and remaining token counts. Each loop: add arrivals, fill active up to capacity, record active ids, decrement remaining, remove completed, then advance time.",
    "hint_zh": "维护 `time`、`waiting`、`active` 和剩余 token。每轮：加入到达请求，填满 active，记录 active id，剩余 token 减一，移除完成请求，然后时间加一。",
    "tests": [
        {
            "name": "Static batch when all arrive together",
            "code": """
requests = [(0, 2), (0, 1), (0, 3)]
schedule = {fn}(requests, max_batch_size=3)
assert schedule == [[0, 1, 2], [0, 2], [2]], f'Got {schedule}'
""",
        },
        {
            "name": "Admits new request after one finishes",
            "code": """
requests = [(0, 2), (0, 1), (1, 2)]
schedule = {fn}(requests, max_batch_size=2)
# step0: 0 and 1 active, request 1 finishes
# step1: request 2 can enter while request 0 continues
assert schedule == [[0, 1], [0, 2], [2]], f'Got {schedule}'
""",
        },
        {
            "name": "Waits for future arrivals without empty steps",
            "code": """
requests = [(2, 1), (4, 2)]
schedule = {fn}(requests, max_batch_size=1)
assert schedule == [[0], [1], [1]], f'Got {schedule}'
""",
        },
        {
            "name": "FIFO admission under capacity pressure",
            "code": """
requests = [(0, 2), (0, 2), (0, 1), (1, 1)]
schedule = {fn}(requests, max_batch_size=2)
assert schedule == [[0, 1], [0, 1], [2, 3]], f'Got {schedule}'
""",
        },
        {
            "name": "Zero-token requests are ignored",
            "code": """
requests = [(0, 0), (0, 1), (1, 0), (1, 1)]
schedule = {fn}(requests, max_batch_size=2)
assert schedule == [[1], [3]], f'Got {schedule}'
""",
        },
        {
            "name": "Active requests are not preempted by lower ids",
            "code": """
requests = [(2, 1), (0, 3)]
schedule = {fn}(requests, max_batch_size=1)
# Request 1 starts at step 0 and should keep the only slot until it finishes.
# Later request 0 waits even though its id is smaller.
assert schedule == [[1], [1], [1], [0]], f'Got {schedule}'
""",
        },
    ],
    "solution": '''def continuous_batching_schedule(requests, max_batch_size):
    remaining = {i: tokens for i, (_, tokens) in enumerate(requests) if tokens > 0}
    pending = sorted(
        [(arrival, i) for i, (arrival, tokens) in enumerate(requests) if tokens > 0]
    )
    waiting = []
    active = []
    schedule = []
    time = 0
    ptr = 0

    while ptr < len(pending) or waiting or active:
        if not active and not waiting and ptr < len(pending) and time < pending[ptr][0]:
            time = pending[ptr][0]

        while ptr < len(pending) and pending[ptr][0] <= time:
            waiting.append(pending[ptr][1])
            ptr += 1

        while waiting and len(active) < max_batch_size:
            active.append(waiting.pop(0))

        if active:
            schedule.append(sorted(active))
            completed = []
            for request_id in active:
                remaining[request_id] -= 1
                if remaining[request_id] == 0:
                    completed.append(request_id)
            active = [request_id for request_id in active if request_id not in completed]
        time += 1

    return schedule''',
    "demo": """requests = [(0, 2), (0, 1), (1, 2)]
print(continuous_batching_schedule(requests, max_batch_size=2))""",
}
