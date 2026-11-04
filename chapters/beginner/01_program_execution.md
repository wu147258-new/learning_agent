# 从一个命令开始：计算机如何运行程序

## 一、为什么输入一行命令就会得到结果

在电脑上双击图标、运行 Python 文件或启动一个网页服务，看起来是三种不同的操作，实际上都涉及同一个问题：操作系统怎样找到程序、让处理器执行指令，并把执行结果交还给用户？

### 1. 程序与进程不是一回事

**程序**是保存在磁盘上的指令及相关数据，例如一个 Python 文件。**进程**是程序的一次运行活动，具有自己的进程标识、内存区域、打开的文件和运行状态。

同一个程序可以同时启动多次，例如同时打开两个编辑器窗口，它们通常对应不同的进程。进程可能成功结束，也可能因为权限不足、文件不存在或内部异常而失败。

一个程序有三种常用的文字交互通道：标准输入（stdin）从终端或其他程序读入数据；标准输出（stdout）输出正常结果；标准错误（stderr）输出错误和诊断信息。

程序结束时一般还会给出**退出状态码**。惯例上，0 表示成功，非 0 表示某种失败，但每个程序对非零状态码的定义可能不同。

### 2. 第一项实验：写下并运行一条指令

先新建名为 hello.py 的纯文本文件，输入：

~~~python
print("Hello, Agent!")
~~~

在 Windows PowerShell 或 Linux/macOS 终端进入该文件所在目录后运行：

~~~text
python hello.py
~~~

若系统中 Python 的命令名称为 python3，则使用：

~~~text
python3 hello.py
~~~

预计的输出是：

~~~text
Hello, Agent!
~~~

这条指令至少涉及三个步骤：终端查找名为 python 的可执行程序；Python 解释器读取 hello.py；解释器调用输出函数，将字符串发送到标准输出。

若收到“找不到 python”之类的提示，说明错误发生在启动解释器阶段，而不是 hello.py 内部。先检查 Python 是否安装、命令名称是否正确，以及可执行程序是否处于操作系统可搜索的位置。

### 3. 如何区分运行失败和错误输出

尝试下面的程序：

~~~python
import sys

print("普通输出")
print("诊断信息", file=sys.stderr)
sys.exit(7)
~~~

它会向两个输出通道分别写入文字，并使用状态码 7 结束。此时虽然终端显示了文字，程序却不属于成功执行。

在 PowerShell 中，程序运行后可以查看退出码：

~~~powershell
python hello.py
$LASTEXITCODE
~~~

在 Linux/macOS 的 bash 中：

~~~bash
python3 hello.py
echo $?
~~~

不要因为屏幕上出现“成功”两个字就认定任务成功。程序真正的退出码、产生的文件以及外部测试结果通常比模型生成的一句自然语言更可靠。

## 二、文件、路径和当前工作目录

假设程序要读取 data/tasks.json。它为什么有时能正常工作，有时却提示找不到文件？

### 1. 文件路径是位置描述，不是文件本身

假设有如下目录：

~~~text
agent-lab/
├── hello.py
├── data/
│   └── tasks.json
└── results/
~~~

**相对路径** data/tasks.json 是相对于进程当前工作目录解释的；**绝对路径** 则包含足以定位文件的完整路径信息。在 Windows 上，绝对路径可能以盘符开头；在 Linux 上通常从 / 开始。

两个容易混淆的位置是程序文件所在目录，以及启动进程时所处的工作目录。从一个完全不同的目录启动 python /somewhere/agent-lab/hello.py，工作目录未必自动切换到 agent-lab。

### 2. 使用 pathlib 构建跨平台路径

~~~python
from pathlib import Path

script_dir = Path(__file__).resolve().parent
task_file = script_dir / "data" / "tasks.json"

print("脚本目录：", script_dir)
print("任务文件：", task_file)
print("任务文件是否存在：", task_file.exists())
~~~

这里的 / 是 pathlib 定义的路径拼接运算，不是数学除法。使用路径对象可以减少自己手写反斜杠和正斜杠造成的错误。

如果任务文件必须存在，应该明确检查：

~~~python
if not task_file.is_file():
    raise FileNotFoundError(f"未找到任务数据：{task_file}")
~~~

**exists()** 和 **is_file()** 不完全相同：目录也可能存在，但它不是普通文件。对文件路径进行授权检查时，还必须考虑符号链接、路径规范化、并发修改等因素；简单字符串判断不能构成安全隔离。

## 三、命令行、环境变量和运行环境

### 1. 终端与命令解释器

终端是供用户输入、显示文字的界面；PowerShell、bash 等命令解释器负责解析用户输入，启动程序并连接输入输出。

Windows 的 PowerShell 与 Linux 的 bash 不是同一门语言。例如在 PowerShell 中查看当前目录可以输入 Get-Location，在 bash 中则使用 pwd。复制命令时必须先确认目标终端。

| 目的 | Windows PowerShell | Linux/macOS bash |
|:--|:--|:--|
| 当前目录 | Get-Location | pwd |
| 列出文件 | Get-ChildItem | ls |
| 切换目录 | Set-Location 路径 | cd 路径 |
| 创建目录 | New-Item -ItemType Directory demo | mkdir demo |
| Python版本 | python --version | python3 --version |

### 2. 环境变量和 PATH

环境变量是进程启动时可以获得的一组键值数据。例如 PATH 通常告诉命令解释器到哪些目录寻找可执行程序，HOME 或 USERPROFILE 通常用于表示用户主目录。

在 PowerShell 中：

~~~powershell
$env:PATH
~~~

在 bash 中：

~~~bash
echo "$PATH"
~~~

如果同一台电脑装有多个 Python，命令 python 指向的解释器可能与你预期的不同。下面的程序可以准确显示当前解释器：

~~~python
import sys
print(sys.version)
print(sys.executable)
~~~

安装 Python 包时，优先使用解释器自身启动包管理器：

~~~text
python -m pip --version
~~~

这样更容易确认运行 pip 的解释器与执行代码的解释器一致。

### 3. 为什么要创建虚拟环境

假设项目甲要求某个库的版本为1.x，项目乙要求2.x。如果都安装到同一个全局Python环境，就可能产生冲突。

虚拟环境为项目提供相对独立的包安装位置。进入项目目录后运行：

~~~text
python -m venv .venv
~~~

PowerShell 激活环境的典型命令：

~~~powershell
.\.venv\Scripts\Activate.ps1
~~~

Linux/macOS bash：

~~~bash
source .venv/bin/activate
~~~

若 PowerShell 阻止激活脚本运行，可直接使用虚拟环境里的 python.exe，而不是为了运行一个脚本就永久放宽全系统执行策略。

虚拟环境**不等于安全沙箱**。它隔离的是包环境，不是文件系统、网络权限或操作系统资源。运行不可信的Agent工具仍需额外隔离。

## 四、把程序变成可验证的实验

### 1. 运行结果必须有预期

一个实验不能只说明“运行一下看看”。它需要输入、预期输出和失败条件。假设目标是计算两个整数的和：

~~~python
def add(a, b):
    return a + b

assert add(2, 3) == 5
assert add(-1, 1) == 0
print("PASS")
~~~

如果断言失败，Python将抛出异常，程序通常以非零状态退出。这比人工检查一段自然语言更适合自动化回归测试。

### 2. 为什么保存版本信息

假设昨天的程序测试通过，今天修改了一行代码后测试失败。能否说明问题由新代码造成？至少还需要知道：Python是否升级、依赖是否改变、输入数据是否变动，以及是否复用了旧缓存。

最小的实验记录应包含代码版本、解释器版本、输入、运行命令、预期结果和实际结果。初学者暂时不必掌握全部版本控制细节，但需要养成对比时保持条件一致的习惯。

### 3. 第一个可复现命令行程序

仓库中的 **examples/beginner_cli.py** 提供了完整的命令行示例，能够读取两个整数，计算结果，并分别处理合法输入与非法输入。

运行：

~~~text
python examples/beginner_cli.py add 12 30
python examples/beginner_cli.py add 12 hello
python examples/beginner_cli.py self-test
~~~

第一条应输出 42；第二条应使用非零退出码提示整数解析错误；第三条应输出测试通过。这三种情况共同构成一个最小的命令行软件验收。

## 五、从普通程序到Agent执行器

一个能够输出文字的程序，不一定能完成用户要求的任务。假设程序要检查项目里是否存在指定文件，还需要明确：它可以读取哪些目录；用户请求如何转换成受限的程序动作；文件不存在时如何报告失败；检查结果如何由程序而不是语言模型的自述确认。

一个可控的工具执行器可表示为：

~~~text
用户请求 → 解析为结构化动作 → 检查类型和参数 → 检查授权
       → 执行操作 → 收集退出码和文件变化 → 独立验证结果
~~~

**解释操作**与**批准操作**是两回事。模型可以建议某个动作，但是否允许执行应由外部系统决定。这一区别贯穿自动编程、安全审计和长期运行的Agent。

## 六、练习与故障排查

### 小节练习

1. 说明程序、进程、线程三个概念中，哪一个表示保存在磁盘上的指令集合。线程与进程有什么区别？
2. 为什么相同的 Python 文件，从两个不同目录启动时可能读取到不同的数据文件？
3. 运行一个程序，屏幕显示“成功”，但退出状态码为3。自动化测试应将它判定为成功还是失败？需要哪些例外规则？
4. 用 pathlib 生成当前脚本同级目录下的 results/result.json 路径。
5. 说明虚拟环境为什么不能替代安全沙箱。

### 实验验收

执行 **python examples/beginner_cli.py self-test**，预期显示 PASS。尝试把 **add 12 hello** 的退出状态码记录下来，并解释这种错误发生在读取输入、数值计算还是输出文件阶段。

\begin{exercisebox}
**排查练习。** 某任务在本地运行正常，在CI中报 FileNotFoundError。已知输入数据确实随项目提交了。请提出至少三项检查，并给出逐项确认的命令或Python语句。
\end{exercisebox}
