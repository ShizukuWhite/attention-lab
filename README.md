# Attention Lab

Attention Lab 是一个交互式 Transformer 学习项目。我们会用小规模实验把架构、数学、Tensor Shape 和数据流逐步呈现出来，让每一步都能观察和检查。JavaScript 与网页代码只是实现工具，不是本项目的学习重点。

## 在本地运行

需要 Node.js 24 或更高版本；本阶段使用 Node.js 24.19.0 编写和检查。无需安装 npm 包、框架或构建工具。

克隆仓库后，在仓库根目录打开终端并运行：

```text
git clone https://github.com/ShizukuWhite/attention-lab.git
cd attention-lab
node server.mjs
```

然后访问 <http://127.0.0.1:3000>。在 Windows 上也可以双击 `start.cmd`。启动脚本会先查找 PATH 中的 Node.js，再尝试当前用户的 Codex Node.js 运行时缓存；如果都找不到，它会说明需要安装 Node.js 24 或将 `node` 加入 PATH。服务只监听本机地址，并且只提供页面需要的静态文件。回到运行服务的终端按 `Ctrl+C` 即可停止。

运行分词契约测试（同样在仓库根目录执行）：

```text
node --test
```

仓库根目录就是 `attention-lab/`，无需进入嵌套目录。`package.json` 也定义了 `start` 和 `test` 脚本；日常运行和检查直接使用上面的 Node 命令即可。

## 当前阶段：Phase 1 — Token Playground

输入文本并点击 **Analyze**，页面会显示 Token 序列、序列长度 `N`、Shape `[N]` 以及每个 Token 的位置。点击一个 Token，可以查看它的文本、从 0 开始的位置索引、从 1 开始的序位和 Unicode 码点数。修改输入后，页面会明确标出当前结果仍来自上一次分析；再次点击 Analyze 会更新结果。

本阶段只采用一种规则：**教学用空白分词**。每段连续的非空白文本算作一个 Token；空格、Tab 和换行用于分隔；标点保留在原片段中；大小写和文本原样保留。例如：

| 输入 | Token 序列 |
| --- | --- |
| `I love my gecko` | `I` · `love` · `my` · `gecko` |
| `hello, world!` | `hello,` · `world!` |
| `我喜欢壁虎` | `我喜欢壁虎` |
| `我 喜欢 壁虎` | `我` · `喜欢` · `壁虎` |

真实语言模型通常使用子词 Tokenizer 和训练得到的词表，切分结果可能与这里不同。本页面不会生成模型 Token ID。

`Sequence Length N` 表示 Token 的位置数量。`[N]` 只说明当前文本列表的长度，不是 Embedding 或数值 Tensor。重复文本仍占据不同位置：`go go` 中两个 `go` 分别位于索引 0 和 1。索引从 0 开始；日常表达中的“第几个”从 1 开始。

## 动手试一试

1. 从 `I love my gecko` 开始，点击 `gecko`。它位于索引 3，也就是第 4 个位置，序列长度为 4。
2. 在文本开头、结尾和单词之间增删空格、Tab 或换行，观察 Token 数量怎样变化。再试 `go go`，看看相同文本如何占据两个位置。
3. 对照 `hello, world!`、`我喜欢壁虎` 和 `我 喜欢 壁虎`。标点仍附着在片段上；在本阶段的规则下，没有空格的中文整段算作一个 Token。

Unicode 码点数不等于屏幕上看到的符号数量。有些组合字符或 Emoji 会由多个码点组成。

## 项目目录

```text
attention-lab/
├── index.html
├── package.json
├── README.md
├── server.mjs
├── start.cmd
├── src/
│   ├── app.js
│   ├── styles.css
│   └── tokenizer.js
└── tests/
    └── tokenizer.test.js
```

建议先读 `src/tokenizer.js`。其中的纯函数 `analyzeText` 按教学用空白规则切分文本，并返回 Token 文本、位置、码点数、序列长度和 Shape；它不读取网页，也不访问网络。

其余文件负责让实验可以操作：`index.html` 提供页面结构，`src/app.js` 连接控件、学习函数和结果显示，`src/styles.css` 设置页面布局，`server.mjs` 提供本地静态服务，`tests/tokenizer.test.js` 固定分词规则以便回归检查。现阶段可以先把网页、样式和服务器代码当作工程支撑，不必深入研究它们。

## 学习路线图

当前只实现 Phase 1。后续阶段计划如下：

1. **Phase 2 — Embedding 可视化：** 为 Token 展示很小的教学向量及其 Shape。
2. **Phase 3 — 向量与点积：** 选择两个向量，逐项查看乘法和求和过程。
3. **Phase 4 — Q、K、V：** 引入投影矩阵，并检查每一步的 Shape。
4. **Phase 5 — Attention Score：** 计算 `QKᵀ`，理解为什么 N 个 Token 会产生 `N × N` 矩阵。
5. **Phase 6 — Scaled Dot-Product Attention：** 分步展示缩放、Softmax、权重和加权求和。
6. **Phase 7 — Attention Heatmap：** 按 Query 和 Key 位置探索实际权重。
7. **Phase 8 — Multi-Head Attention：** 比较不同 Head，查看拼接与输出投影。
8. **Phase 9 — Transformer Block：** 跟踪 Attention、残差连接、归一化和前馈网络。
9. **Phase 10 — 位置信息：** 比较不同顺序的序列，观察加入位置信息后的变化。
10. **Phase 11 — Mask：** 可视化 Padding Mask 与 Causal Mask。
11. **Phase 12 — Transformer 架构：** 串起 Encoder 和 Decoder 模块及其 Shape。
12. **Phase 13 — TensorFlow：** 将框架实现与此前手动拆解的运算对照。
13. **Phase 14 — Tiny Transformer：** 观察一个小模型的预测、Loss、Gradient 和权重更新。

每个新概念优先按“直觉 → 数学 → Shape → 代码”的顺序学习。前期会使用少量 Token 和小矩阵，使中间数值容易检查。框架语法是后续对照已有概念的工具，不会变成独立的 JavaScript 或 Python 入门课程。
