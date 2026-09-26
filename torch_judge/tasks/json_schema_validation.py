"""JSON-schema validation for tool arguments."""

TASK = {
    "title": "Validate Tool Arguments Against a Schema",
    "title_zh": "按 JSON Schema 校验工具参数",
    "difficulty": "Medium",
    "description_en": "Validate function-calling arguments against a JSON-schema subset — the check every tool-serving backend (and every Qwen/Llama agent harness) runs before invoking a tool.\n\nSupport the keywords: `type`, `properties`, `required`, `enum`, `items`. Types map to Python: `object`=dict, `array`=list, `string`=str, `integer`=int, `number`=int or float, `boolean`=bool, `null`=None. Note that in Python `bool` is a subclass of `int` — a boolean must NOT satisfy `integer` or `number`.\n\n**Signature:** `validate_tool_arguments(args, schema) -> list[str]`\n\n**Parameters:**\n- `args` — the parsed arguments object (usually the `arguments` dict of a tool call)\n- `schema` — JSON schema with root type `object`\n\n**Returns:** list of error strings, empty when valid. Error formats (exact):\n- type: `<path> must be of type <expected>, got <actual>`\n- required: `<obj_path> is missing required property '<key>'`\n- enum: `<path> must be one of [<v1>, <v2>, ...]` (JSON list)\n\nPaths are `$` for the root, `$.city` for properties, `$.address.city` nested, `$.tags[0]` for array items.\n\n**Constraints:**\n- On a type error, do not recurse into that value further\n- `integer` accepts ints only; `number` accepts ints and floats; booleans satisfy neither",
    "description_zh": "按 JSON Schema 子集校验函数调用参数——这是所有工具服务后端（以及每个 Qwen/Llama agent 框架）在真正调用工具前跑的检查。\n\n支持关键字：`type`、`properties`、`required`、`enum`、`items`。类型映射到 Python：`object`=dict、`array`=list、`string`=str、`integer`=int、`number`=int 或 float、`boolean`=bool、`null`=None。注意 Python 里 `bool` 是 `int` 的子类——布尔值不能算 `integer` 或 `number`。\n\n**签名:** `validate_tool_arguments(args, schema) -> list[str]`\n\n**参数:**\n- `args` — 解析后的参数对象（通常是工具调用的 `arguments` dict）\n- `schema` — 根类型为 `object` 的 JSON Schema\n\n**返回:** 错误字符串列表，合法时为空。错误格式（精确）：\n- 类型：`<path> must be of type <expected>, got <actual>`\n- 必填：`<obj_path> is missing required property '<key>'`\n- 枚举：`<path> must be one of [<v1>, <v2>, ...]`（JSON 列表）\n\n路径：根为 `$`，属性 `$.city`，嵌套 `$.address.city`，数组元素 `$.tags[0]`。\n\n**约束:**\n- 类型错误后不再深入该值\n- `integer` 只接受 int；`number` 接受 int 和 float；布尔两者都不算",
    "function_name": "validate_tool_arguments",
    "hint": "Recursive walk carrying the path string. Map Python types to schema names first, and special-case `bool` before `int` since `isinstance(True, int)` is True.",
    "hint_zh": "递归遍历并携带路径字符串。先把 Python 类型映射为 schema 名；注意 `isinstance(True, int)` 为真，所以要先判 `bool` 再判 `int`。",
    "tests": [
        {
            "name": "Valid arguments produce no errors",
            "code": """
schema = {
    'type': 'object',
    'properties': {
        'city': {'type': 'string'},
        'days': {'type': 'integer'},
        'unit': {'enum': ['celsius', 'fahrenheit']},
    },
    'required': ['city'],
}
args = {'city': 'Beijing', 'days': 3, 'unit': 'celsius'}
errors = {fn}(args, schema)
assert errors == [], f'Expected no errors, got: {errors}'
""",
        },
        {
            "name": "Missing required property",
            "code": """
schema = {'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']}
errors = {fn}({}, schema)
assert errors == ["$ is missing required property 'city'"], f'Got: {errors}'
""",
        },
        {
            "name": "Nested property type error",
            "code": """
schema = {
    'type': 'object',
    'properties': {
        'address': {
            'type': 'object',
            'properties': {'zip': {'type': 'string'}},
            'required': ['zip'],
        },
    },
}
errors = {fn}({'address': {'zip': 10086}}, schema)
assert errors == ['$.address.zip must be of type string, got integer'], f'Got: {errors}'
""",
        },
        {
            "name": "Enum violation",
            "code": """
schema = {'type': 'object', 'properties': {'unit': {'enum': ['celsius', 'fahrenheit']}}, 'required': ['unit']}
errors = {fn}({'unit': 'kelvin'}, schema)
assert errors == ['$.unit must be one of ["celsius", "fahrenheit"]'], f'Got: {errors}'
""",
        },
        {
            "name": "Boolean is not a number or integer",
            "code": """
schema = {
    'type': 'object',
    'properties': {
        'count': {'type': 'integer'},
        'ratio': {'type': 'number'},
    },
}
errors = {fn}({'count': True, 'ratio': False}, schema)
assert len(errors) == 2, f'Expected 2 errors, got: {errors}'
assert '$.count must be of type integer, got boolean' in errors
assert '$.ratio must be of type number, got boolean' in errors
""",
        },
        {
            "name": "Array item validation",
            "code": """
schema = {
    'type': 'object',
    'properties': {
        'tags': {'type': 'array', 'items': {'type': 'string'}},
    },
}
errors = {fn}({'tags': [1, 'a', 2.5]}, schema)
assert errors == ['$.tags[0] must be of type string, got integer',
                  '$.tags[2] must be of type string, got number'], f'Got: {errors}'
""",
        },
        {
            "name": "Root must be an object",
            "code": """
schema = {'type': 'object', 'properties': {}, 'required': []}
errors = {fn}([1, 2, 3], schema)
assert errors == ['$ must be of type object, got array'], f'Got: {errors}'
""",
        },
        {
            "name": "Integer satisfies number, no recursion past type error",
            "code": """
schema = {
    'type': 'object',
    'properties': {
        'ratio': {'type': 'number'},
        'blob': {'type': 'string'},
    },
}
errors = {fn}({'ratio': 2, 'blob': {'nested': 1}}, schema)
assert '$.ratio' not in ' '.join(errors), 'int should satisfy number'
assert len([e for e in errors if e.startswith('$.blob')]) == 1, f'Got: {errors}'
""",
        },
    ],
    "solution": '''import json

_TYPE_NAMES = (dict, "object"), (list, "array"), (str, "string"), (bool, "boolean"), (int, "integer"), (float, "number"), (type(None), "null")


def _actual_type(value):
    # bool must be tested before int: isinstance(True, int) is True
    for py, name in _TYPE_NAMES:
        if isinstance(value, py) and not (py is int and isinstance(value, bool)):
            if py is int and isinstance(value, bool):
                continue
            return name
    return "null"


def _matches(value, expected):
    a = _actual_type(value)
    if a == expected:
        return True
    if expected == "number" and a == "integer":
        return True
    return False


def validate_tool_arguments(args, schema):
    errors = []

    def walk(value, sub, path):
        expected = sub.get("type")
        if expected is not None and not _matches(value, expected):
            errors.append(f"{path} must be of type {expected}, got {_actual_type(value)}")
            return
        if "enum" in sub and value not in sub["enum"]:
            errors.append(f"{path} must be one of {json.dumps(sub['enum'])}")
            return
        actual = _actual_type(value)
        if actual == "object":
            for key in sub.get("required", []):
                if key not in value:
                    errors.append(f"{path} is missing required property '{key}'")
            for key, prop in sub.get("properties", {}).items():
                if key in value:
                    walk(value[key], prop, f"{path}.{key}")
        elif actual == "array" and "items" in sub:
            for i, item in enumerate(value):
                walk(item, sub["items"], f"{path}[{i}]")

    walk(args, schema, "$")
    return errors''',
    "demo": """schema = {
    'type': 'object',
    'properties': {
        'city': {'type': 'string'},
        'days': {'type': 'integer'},
        'unit': {'enum': ['celsius', 'fahrenheit']},
    },
    'required': ['city', 'unit'],
}
for args in ({'city': 'Beijing', 'unit': 'celsius'}, {'days': True, 'unit': 'kelvin'}):
    print(args, '->', validate_tool_arguments(args, schema))""",
}
