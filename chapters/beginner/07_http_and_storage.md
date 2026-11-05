# HTTP、API与数据库：程序怎样与外部世界交换数据

## 互联网请求为什么不只是调用一个函数

在本地Python程序中，函数调用通常在同一进程内完成；而当模型需要从远程服务读取数据，操作可能涉及DNS解析、TCP连接、TLS加密、服务器鉴权和网络超时。

### 域名、IP地址与端口

域名用于人类识别服务，例如 example.org。DNS可将域名解析到相应的网络地址；IP地址用于网络层寻址；端口号用于在主机上区分不同的传输端点。

URL可以包含以下部分：

~~~text
https://example.org:443/api/tasks?status=running
  │        │       │      │          │
 协议     域名     端口    路径       查询参数
~~~

不是所有URL都会显式写出端口。HTTPS服务通常使用443端口，但实际部署可能不同。URL并不证明服务安全，访问前仍要检查目标、权限及证书验证结果。

### HTTP请求与响应

典型HTTP请求由方法、目标路径、首部和可选的请求体组成：

~~~http
GET /api/tasks?status=running HTTP/1.1
Host: example.org
Accept: application/json
~~~

服务器可能返回：

~~~http
HTTP/1.1 200 OK
Content-Type: application/json

{"tasks":[{"id":"t-001","status":"running"}]}
~~~

HTTP状态码属于协议响应的一部分。200通常表示请求成功，但不能据此认定业务目标已经完成。例如创建任务返回成功，只能证明服务接受了请求，并不代表异步任务已经运行结束。

## GET、POST和重复请求的差异

### 读操作与写操作

GET一般用于获取资源，POST经常用于提交处理请求。PUT、PATCH、DELETE分别用于不同的资源更新或删除语义。

HTTP方法具有标准约定，但真实系统的服务器实现未必遵守这些约定。因此Agent不能只看到“GET”就断言绝对无副作用，也不能只看到“POST”就认定一定创建了新对象。

### 什么是幂等

如果一个操作重复执行与执行一次具有相同的**预期最终状态**，可称为幂等。例如将某任务状态设置为已暂停，在有效实现下重复设置通常不改变最终状态。

但是发送一条邮件两次往往会产生两条邮件，属于不同的效果。当网络请求发生超时，客户端可能不知道服务器是否已经完成写入，因此不能无条件重试。

一种常见方案是为创建请求指定唯一的幂等标识符，服务器记录已经处理过的标识符，在重试时返回先前结果。这依赖服务器明确实现，而不是只在请求中加入任意字段。

## JSON与API结构化数据

### API请求中最容易发生的类型错误

假设服务定义如下JSON：

~~~json
{
  "task_id": "t-001",
  "retries": 3,
  "enabled": true
}
~~~

retries应是整数，不应该传入字符串"3"；enabled应是布尔值，而不是字符串"true"。

在模型调用工具时，生成一段可以解析的JSON只是第一层要求；还需要字段存在性、数值范围、权限、超时与数据一致性检查。

### 使用Python解析URL

~~~python
from urllib.parse import urlsplit, parse_qs

parts = urlsplit(
    "https://example.org/api/tasks?status=running"
)
assert parts.scheme == "https"
assert parts.hostname == "example.org"
assert parts.path == "/api/tasks"
assert parse_qs(parts.query)["status"] == ["running"]
~~~

urlsplit只负责解析字符串，不会验证服务是否可信。真实应用不能因URL被成功解析就允许它访问内部网络或云元数据端点。

## 数据库：为什么内存里的字典不够用

### 进程退出后信息去哪了

如果程序用Python字典记录任务状态，进程退出后内存数据通常无法直接恢复。长期运行的Agent需要把重要任务记录持久化：任务ID、已完成步骤、版本、剩余预算和最终检查结果。

数据库提供持久存储、查询、事务处理等能力。教学实验使用Python标准库自带的SQLite，它可以存储在文件中，也可以使用内存数据库。

### 建立第一个关系表

~~~python
import sqlite3

db = sqlite3.connect(":memory:")
db.execute(
    "CREATE TABLE tasks ("
    "task_id TEXT PRIMARY KEY,"
    "status TEXT NOT NULL,"
    "attempts INTEGER NOT NULL DEFAULT 0)"
)
db.execute(
    "INSERT INTO tasks(task_id,status,attempts) "
    "VALUES (?, ?, ?)",
    ("t-001", "pending", 0),
)
row = db.execute(
    "SELECT status,attempts FROM tasks WHERE task_id = ?",
    ("t-001",),
).fetchone()
assert row == ("pending", 0)
db.close()
~~~

这里的问号是参数占位符。不要为了简单把用户输入直接拼入SQL字符串；使用参数化查询可避免将数据误当作SQL语法执行。

### 主键与数据一致性

task_id被定义为主键，意味着同一张表里不能插入两条具有相同任务ID的记录。这是一项由数据库保证的约束，比仅靠模型提示“请不要创建重复任务”更可靠。

此外，一个任务的状态变化可能需要满足明确规则，例如从pending转为running，再转为succeeded或failed。不能允许无条件地把任意状态改成任意状态。

## 事务：多个修改为什么要作为整体考虑

### 部分成功会造成什么问题

假设完成任务需要同时修改任务状态、扣减可用预算、添加操作日志。若第一步成功、第二步失败、第三步未执行，记录可能彼此矛盾。

数据库事务把一组操作作为一个逻辑单位。典型关系数据库会提供原子性、一致性、隔离性和持久性相关保证（ACID），但这些保证的精确含义和边界依具体数据库、配置与事务隔离级别而定。

Python sqlite3 中可用事务上下文：

~~~python
import sqlite3

db = sqlite3.connect(":memory:")
db.execute("CREATE TABLE counters (id INTEGER PRIMARY KEY, value INTEGER)")
db.execute("INSERT INTO counters VALUES (1, 0)")
with db:
    db.execute("UPDATE counters SET value=value+1 WHERE id=1")
assert db.execute("SELECT value FROM counters WHERE id=1").fetchone()[0] == 1
db.close()
~~~

在事务中发生异常时，未提交的修改通常需要回滚。但数据库事务无法自动撤销已经发送的电子邮件或已经调用的第三方API。

### 并发任务如何避免相互覆盖

两个Agent同时读取某任务版本号10，各自产生修改。若第一个Agent提交后版本号变成11，第二个Agent仍用旧版本覆盖数据，就可能丢失第一个修改。

可以使用条件更新：

~~~sql
UPDATE tasks
SET status = 'done', version = version + 1
WHERE task_id = 't-001' AND version = 10;
~~~

执行后必须检查受影响行数。如果为0，说明记录不存在或版本已变化，不能直接宣称修改完成。

## 从API、数据库走向Agent运行时

一个长任务的执行过程通常包含：

1. 接收用户目标并创建任务记录；
2. 读取允许使用的工具清单；
3. 根据观察生成一个待执行动作；
4. 验证权限和参数；
5. 发送外部请求并记录请求标识符；
6. 将确认的结果写入数据库；
7. 恢复失败任务时判断哪些步骤可能已经执行。

如果第5步的请求超时，第6步没有完成，Agent下一次启动时不能简单再次发送同样的写操作。系统应查询是否有外部完成记录，再决定如何恢复。

从这里开始，Agent已经不仅是文本模型，而是包含通信、数据存储、状态转移和故障恢复的软件系统。

## 实验与练习

### 离线实验

执行 **python examples/foundation_http_sql.py**。该实验不会访问互联网，而是使用URL解析和SQLite内存数据库验证：

- URL的协议、主机和查询参数；
- 参数化SQL插入和检索；
- 重复主键的拒绝；
- 对版本号不匹配的并发修改拒绝；
- 事务出现异常时的回滚。

### 思考题

1. HTTP 200能否证明异步任务已经完成？为什么？
2. 客户端发起付款请求后发生超时，应该如何区分“服务器尚未处理”和“服务器已经处理但响应丢失”？
3. 参数化查询与手工拼接SQL字符串的安全性质有何不同？
4. 若两个Agent同时修改一条数据库记录，应当把版本比较放在模型提示词里还是数据库条件更新里？为什么？
5. 为什么一个成功解析的URL也可能成为危险工具输入？
6. 事务可以回滚本地数据库修改，却不能自动撤销哪些外部副作用？请举两个例子。

\begin{exercisebox}
\textbf{项目题。} 设计一个数据库表保存任务的唯一ID、状态、版本号、重试次数和最近一次外部请求标识符。实现“只有当前版本符合预期时才能更新”的操作，构造两个执行者读取同一旧版本的冲突实验。
\end{exercisebox}
