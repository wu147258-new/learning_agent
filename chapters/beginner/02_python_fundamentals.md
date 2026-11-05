# Python基础：数据、函数和可验证的程序

## 从一个输入到一个输出

写程序的第一步并不是记住几十个语法关键字，而是想清楚：数据从哪里来，经过什么操作，最后输出什么结果。

### 表达式、变量和赋值

下面的代码计算三个任务的平均耗时：

~~~python
durations = [2.0, 5.0, 8.0]
total = sum(durations)
average = total / len(durations)
print(average)
~~~

输出为 5.0。变量是程序中对对象的引用名称，赋值语句将右侧表达式的计算结果绑定到左侧名称。Python变量不需要提前声明类型，但运行时对象有类型。

~~~python
count = 3                # int：整数
rate = 0.75              # float：浮点数
name = "test"            # str：字符串
passed = True            # bool：布尔值
value = None             # NoneType：没有给出值
~~~

数值3、字符串"3"、布尔值True和None不是同一对象。尤其在JSON工具调用中，类型错误是实际故障的常见来源。

### 为什么浮点数有时不等于十进制手算值

计算机中的常见浮点数以二进制有限精度表示。十进制数0.1一般不能被有限位二进制浮点数精确表示：

~~~python
print(0.1 + 0.2 == 0.3)
~~~

结果通常为 False。这并不意味着计算机无法加法，而是数值的内部近似表示产生了舍入误差。

比较浮点结果时，可以使用：

~~~python
import math

assert math.isclose(0.1 + 0.2, 0.3, abs_tol=1e-12)
~~~

处理货币、计费等需要固定十进制规则的任务时，应考虑 Decimal 或以最小计量单位表示的整数，不能随意比较两个浮点数是否完全相等。

## 条件分支和重复执行

### 条件判断表示明确的逻辑规则

假设系统只允许在确认用户已授权且文件确实存在时读取文件：

~~~python
authorized = True
file_exists = False

if authorized and file_exists:
    print("允许读取")
else:
    print("拒绝或无法读取")
~~~

and 是逻辑与：两个条件都成立才为真。or 是逻辑或；not 用于否定。括号可以明确复杂条件的执行顺序。

### for循环和while循环有什么区别

已知要处理一个有限任务列表时，可使用 for 循环：

~~~python
tasks = ["read", "test", "report"]
for index, task in enumerate(tasks, start=1):
    print(index, task)
~~~

若不知道执行几次才能达到目标，可使用 while 循环；但必须有清晰的停止条件：

~~~python
attempt = 0
max_attempts = 3
while attempt < max_attempts:
    attempt += 1
    print("第", attempt, "次尝试")
~~~

无限执行循环对普通练习只是程序错误；对具备工具权限的Agent，可能意味着重复付款、重复发邮件或无限消耗API费用。因此运行时系统通常还需要执行时间、步骤数和费用上限。

## 函数：将复杂问题分成可验证的部分

### 参数、返回值和副作用

~~~python
def total_cost(prices):
    return sum(prices)

assert total_cost([2, 3, 5]) == 10
~~~

函数名为 total_cost；prices 是形式参数；return 指定返回值。相较于把所有代码写在一起，函数使输入输出关系更明确。

如果一个函数只依赖参数并返回结果，而不读取外部状态，也不修改文件，通常更容易测试。

对比：

~~~python
def add(a, b):
    return a + b
~~~

与：

~~~python
def append_report(text, filename):
    with open(filename, "a", encoding="utf-8") as fp:
        fp.write(text + "\n")
~~~

第二个函数不仅返回计算结果，还**修改外部文件**。这种对外部世界可观察的修改称为副作用。对Agent工具调用而言，必须识别读取类操作和具有副作用的写入操作。

### 输入校验不应交给模型“自行理解”

例如一个工具只允许把两个整数相加：

~~~python
def add_integers(a, b):
    if type(a) is not int or type(b) is not int:
        raise TypeError("Expected integer arguments")
    return a + b
~~~

这里使用 type(a) is int 是为了严格拒绝 bool，因为在Python中 bool 是 int 的子类。若使用 isinstance(True, int)，结果会是 True；这种细节会直接影响结构化工具参数校验。

不允许的输入必须由确定性的程序规则拒绝。不能把“请模型尽量不要填错参数”当作校验机制。

## 列表、字典和集合

### 列表保存有顺序的数据

~~~python
tasks = ["read", "test", "report"]
assert tasks[0] == "read"
assert len(tasks) == 3
tasks.append("archive")
~~~

列表按位置存储，位置从0开始。访问不存在的下标会抛出 IndexError，而不是自动返回空字符串。

### 字典保存键到值的映射

~~~python
task = {
    "id": "t-001",
    "status": "pending",
    "attempts": 0,
}
task["attempts"] += 1
assert task["attempts"] == 1
~~~

字典适合表示有明确字段的任务记录。对于工业系统，还需要定义字段是否必填、可接受的数据类型以及状态变化规则。

集合可用于快速判断某个值是否属于允许集合：

~~~python
allowed = {"read", "test"}
assert "read" in allowed
assert "delete" not in allowed
~~~

但集合的迭代顺序不应被用作业务规则的一部分。

### 组合类型表示一个状态对象

~~~python
state = {
    "task_id": "t-001",
    "history": [
        {"tool": "read", "ok": True},
        {"tool": "test", "ok": False},
    ],
    "completed": False,
}
~~~

这个对象描述的是任务执行过程中的状态，而不是“模型的想法”。如果程序退出后需要继续任务，状态需要保存在可恢复的外部存储中。

## JSON：程序与模型交换数据的常用格式

### JSON为什么适合工具调用

下面是一条结构化请求：

~~~json
{
  "tool": "add",
  "arguments": {
    "a": 12,
    "b": 30
  }
}
~~~

它比“帮我加一下12和30”更容易被程序解析、验证和执行。但**JSON格式正确并不代表操作已获授权**：工具名、参数类型、具体动作仍需要独立检查。

### 在Python中解析JSON

~~~python
import json

request_text = '{"tool":"add","arguments":{"a":12,"b":30}}'
request = json.loads(request_text)
assert request["tool"] == "add"
assert request["arguments"]["a"] == 12
~~~

JSON中字符串使用双引号，布尔值为 true/false，空值为 null。Python对象与JSON数据可以互相转换，但不是所有Python对象都可以直接转换为JSON。

### 文件读写

~~~python
import json
from pathlib import Path

path = Path("task.json")
payload = {"task_id": "t-001", "status": "pending"}
path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
restored = json.loads(path.read_text(encoding="utf-8"))
assert restored == payload
~~~

这里读取和写入都明确指定UTF-8，减少跨平台中文编码问题。正式服务端程序不能直接使用任意用户提供的文件路径。

## 异常与失败分类

### 错误为什么不能全部返回“执行失败”

一个工具执行器至少要区分输入错误、权限错误、文件不存在、外部服务不可用和结果未知。它们的恢复办法不同。

~~~python
def divide(a, b):
    if b == 0:
        raise ValueError("Denominator cannot be zero")
    return a / b

try:
    print(divide(10, 0))
except ValueError as exc:
    print("输入错误：", exc)
~~~

捕获异常的目标是处理**可以预期且可以安全处理**的失败，不是让程序吞掉所有错误继续运行。裸写 except: pass 可能隐藏逻辑错误，并导致外部状态不一致。

### 为什么重试之前必须判断副作用

读文件超时，可以尝试重试；向外部系统转账超时，却不能立即认定“转账没发生”。网络断开时，服务端可能已经处理完成，只是响应没有返回。

此时需要查询外部状态、使用幂等键或执行一致性检查。这是由副作用与不确定性共同导致的工程问题，和提示词写得是否漂亮无关。

## 把知识组合为一个最小工具服务

### 数据验证到操作执行

一个简化的工具执行器可分为四个函数：

~~~python
def parse_request(text):
    import json
    return json.loads(text)

def validate_request(request):
    if request.get("tool") != "add":
        raise ValueError("Unknown tool")
    args = request.get("arguments")
    if not isinstance(args, dict):
        raise TypeError("arguments must be an object")
    if type(args.get("a")) is not int or type(args.get("b")) is not int:
        raise TypeError("a and b must be integers")
    return args["a"], args["b"]

def run_request(request):
    a, b = validate_request(request)
    return a + b

def format_response(result):
    return {"ok": True, "result": result}
~~~

这些函数的职责可以分别测试。如果未来增加删除文件等工具，必须加入与操作类型匹配的授权检查，不能仅扩大允许的工具名称列表。

### 如何验证一个程序真的可靠

除了正常情况，还应测试：

- 请求缺少tool字段；
- arguments字段不是对象；
- a或b是字符串；
- a或b是布尔值；
- 请求包含未知工具；
- 两个大整数相加。

测试失败不意味着程序应当崩溃并造成外部破坏。工具层需要将异常转换为受控错误响应，同时保留足够的诊断信息。

仓库里的 **examples/beginner_cli.py** 包含命令行实现与自检。

## 练习

### 基础题

1. 为什么0.1 + 0.2 == 0.3通常得到False？
2. 在Python中，True与1是否属于相同类型？为什么严格校验整数时需要关注这个细节？
3. 写一个函数count_success(results)，其中results是布尔值列表，返回成功数和任务总数。
4. 为什么把未知异常都写成 except: pass 可能造成真实的系统故障？
5. 给工具请求添加字段 request_id，并说明它在审计和幂等处理中可能有什么用。

### 综合题

实现一个JSON工具执行器，接受两种只读操作：求和与计算均值。要求严格验证参数类型、处理空列表、拒绝未知工具名，使用断言验证至少8种输入。

\begin{exercisebox}
\textbf{讨论。} 一个模型把工具参数 a 写成字符串"12"。执行器应自动把它转换为整数12，还是拒绝？请分别讨论计算器、财务转账和删除文件三种场景下的风险。
\end{exercisebox}
