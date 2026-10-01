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

运行项目测试（同样在仓库根目录执行）：

```text
node --test
```

仓库根目录就是 `attention-lab/`，无需进入嵌套目录。`package.json` 也定义了 `start` 和 `test` 脚本；日常运行和检查直接使用上面的 Node 命令即可。

## 当前阶段：Phase 1–4 — Token、Embedding、点积与 Q/K/V 投影

输入文本并点击 **Analyze**，页面会显示 Token 序列、序列长度 `N`、Token 列表 Shape `[N]` 和每个 Token 的位置。点击一个 Token，可以查看它的文本、从 0 开始的位置索引、从 1 开始的序位、Unicode 码点数及对应向量。页面还会把所有向量按 Token 位置排列为数值矩阵 `X`。Phase 3 允许独立选择两个 Token 的向量，观察逐项乘法、累计和及点积结果；Phase 4 会从当前 `X` 生成 `Q`、`K`、`V`。修改输入后，页面会标出四个阶段仍来自上一次分析；再次点击 Analyze 会同时更新。

### Phase 1：教学用空白分词

每段连续的非空白文本算作一个 Token；空格、Tab 和换行用于分隔；标点保留在原片段中；大小写和文本原样保留。例如：

| 输入 | Token 序列 |
| --- | --- |
| `I love my gecko` | `I` · `love` · `my` · `gecko` |
| `hello, world!` | `hello,` · `world!` |
| `我喜欢壁虎` | `我喜欢壁虎` |
| `我 喜欢 壁虎` | `我` · `喜欢` · `壁虎` |

真实语言模型通常使用子词 Tokenizer 和训练得到的词表，切分结果可能与这里不同。本页面不会生成模型 Token ID。

`Sequence Length N` 表示 Token 的位置数量。Token 列表的 `[N]` 是符号序列的长度；Phase 2 新增的 `X` 才是数值矩阵。重复文本仍占据不同位置：`love love` 中两个 `love` 分别位于索引 0 和 1。索引从 0 开始；日常表达中的“第几个”从 1 开始。

### Phase 2：人工教学 Embedding

每个 Token 会从一个固定查找表中取得 3 个数字。当前教学表是：

| Token | 人工设定的向量 |
| --- | --- |
| `I` | `[0.2, 0.7, 0.1]` |
| `love` | `[0.8, 0.3, 0.4]` |
| `my` | `[0.4, 0.6, 0.2]` |
| `gecko` | `[0.7, 0.9, 0.5]` |

查找区分大小写，也不移除标点，所以 `I` 有表中向量，而 `i` 和 `love!` 是未知 Token。未知 Token 使用 `[0, 0, 0]` 作为占位，并明确标注来源。这不是模型训练或词义推断的结果；两个未知 Token 得到相同占位向量，也不表示它们意义相同。四个表中向量同样是人工指定的示例，三个分量目前没有被赋予“情感”或“动物”等含义。

### Phase 3：向量与点积

选择向量 A 和 B 后，页面显示两个 Token 的位置、Embedding、向量 Shape 和来源。点积把两个向量相同分量位置的数字相乘，再把乘积加起来。同号分量产生正贡献，异号分量产生负贡献；点积可以表示一种数值匹配程度，也会受向量长度影响。这里的 Embedding 是人工指定、未经训练的教学数字，因此结果不是模型学出的词义相似概率。

数学上可以写成：

```text
a · b = Σ a[i] × b[i]
```

默认选择 `I` 与 `gecko`：

```text
0.2 × 0.7 + 0.7 × 0.9 + 0.1 × 0.5
= 0.14 + 0.63 + 0.05
= 0.82
```

每一项展示乘积和截至该项的累计和。两个输入向量的 Shape 都是 `[3]`；逐项相乘得到 3 个乘积 `[3]`，求和后得到一个标量，Shape 为 `[]`。`[]` 表示没有维度轴，不是空数组；结果仍是一个数字。

核心计算按分量循环：

```js
let sum = 0;
for (let i = 0; i < a.length; i += 1) {
  sum += a[i] * b[i];
}
```

乘法产生每一项的贡献，循环把贡献累加到 `sum`。比如 `love · gecko = 1.03`，点积不限制在 0 到 1。页面最多显示 6 位小数，内部运算保留原始精度。

如果选择未知 Token，零结果来自它的零向量占位；这不代表两个词不相关或词义相同。未知 Token 的来源会与数值一起显示。

### Phase 4：Q、K、V 投影

直觉上，同一个 Embedding 矩阵 `X` 分别乘以三组不同权重，得到三种表示。`Q` 是后续发出查询的表示，`K` 是后续被匹配的表示，`V` 是后续被汇总的内容；这些角色要等后续 Attention 运算连接起来才会发挥作用。本阶段只准备 `Q`、`K`、`V`，还没有计算 Token 之间的分数或权重。

页面使用固定、未训练且没有 bias 的小权重矩阵：

| 矩阵 | Shape | 权重 |
| --- | --- | --- |
| `Wq` | `[3, 2]` | `[[1, 0], [0, 1], [1, -1]]` |
| `Wk` | `[3, 2]` | `[[1, 1], [1, -1], [-1, 0]]` |
| `Wv` | `[3, 2]` | `[[1, -1], [1, 1], [0, 1]]` |

每个 Token 行都使用同一组权重：

```text
Q = XWq
K = XWk
V = XWv
```

一个输出格 `Y[i, j]` 由 `X` 的第 `i` 行与 `W` 的第 `j` 列做点积：

```text
Y[i, j] = Σ X[i, t] × W[t, j]
```

这不是对应位置的数字直接相乘；每项使用相同的中间索引 `t`，然后将乘积相加。默认句子的 `X` Shape 是 `[4, 3]`，每个权重 Shape 是 `[3, 2]`，因此 `Q`、`K`、`V` 的 Shape 都是 `[4, 2]`。中间的 3 个分量被求和，保留 4 个 Token 行和 2 个输出列；本例 `d_model = 3`、`d_k = 2`、`d_v = 2`。通常 `d_v` 可以与 `d_k` 不同，这里取相同维度以集中观察投影过程。

默认值可以手算检查：

```text
Q[0, 0] = 0.2 × 1 + 0.7 × 0 + 0.1 × 1 = 0.3
K[0, 1] = 0.2 × 1 + 0.7 × (-1) + 0.1 × 0 = -0.5
```

`K[0, 1]` 中的负贡献来自固定权重 `-1`，不是负概率或负面含义。`V` 也由同一 `X` 与 `Wv` 按相同规则计算。页面保留完整计算精度，显示时最多保留 6 位小数并去掉末尾的零。

### 动手试一试

1. 保持默认的 `I love my gecko`。查看点积面板的三项乘积与累计和，确认 `I · gecko = 0.82`。
2. 将向量 A 改成 `love`，B 保持 `gecko`，观察结果 `1.03`，它可以大于 1。再选择同一个位置两次，查看自点积；例如 `love · love = 0.89`。
3. 在 Phase 4 点击默认选中的 `Q[0, 0]`，逐项检查 `X` 第 0 行与 `Wq` 第 0 列，得到 `0.3`。再点击 `K[0, 1]`，观察负权重带来的负贡献和结果 `-0.5`。
4. 分析 `love love`，从两个不同位置各选一个 `love`。文本和向量数值虽然相同，位置索引仍分别为 0 和 1；Q/K/V 也会保留两行。再试 `i love!`、中文、单个 Token 和空白输入，留意未知 Token 的零向量来源、单 Token 自点积与 `[0, 2]` 的空投影 Shape。

输入新文本但先不点 **Analyze** 时，Tokens、Embedding、点积和 Q/K/V 都仍来自旧序列；点击 **Analyze** 后四者会一起更新。选择 Q/K/V 的输出格不会更改 Token 详情或 Phase 3 的两个点积选择。

Unicode 码点数不等于屏幕上看到的符号数量。有些组合字符或 Emoji 会由多个码点组成。

### 读懂这些 Shape

- Token 列表：`[N]`，表示序列中有 N 个符号位置。
- 单个 Embedding：`[3]`，每个 Token 对应 3 个向量分量；`d_model = 3`。
- 数值矩阵 `X`：`[N, 3]`，每一行对应一个 Token 位置，每一列对应一个分量。空序列的矩阵 Shape 是 `[0, 3]`。
- 每个投影权重 `Wq`、`Wk`、`Wv`：`[3, 2]`；`X [N, 3] × W [3, 2]` 得到 `Q`、`K` 或 `V` 的 `[N, 2]`，中间的 3 被求和。空序列的三个投影 Shape 都是 `[0, 2]`，权重仍为 `[3, 2]`。
- 点击一个 Q/K/V 输出格时，输入行和权重列的 Shape 都是 `[3]`；逐项乘积 Shape 是 `[3]`，求和后的单个输出格是标量 Shape `[]`。
- 点积输出：`[]`，表示没有维度轴的标量 Shape；点积值本身仍是一个数字。

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
│   ├── dot-product.js
│   ├── embedding.js
│   ├── matrix-multiply.js
│   ├── projection.js
│   ├── projection-view.js
│   ├── styles.css
│   └── tokenizer.js
└── tests/
    ├── dot-product.test.js
    ├── embedding.test.js
    ├── matrix-multiply.test.js
    ├── projection.test.js
    └── tokenizer.test.js
```

建议先读 `src/tokenizer.js`、`src/embedding.js`、`src/dot-product.js`、`src/matrix-multiply.js` 和 `src/projection.js`。纯函数 `analyzeText` 按教学用空白规则切分文本，返回 Token 文本、位置、码点数、序列长度和 Shape；`embedTokens` 按固定表给 Token 查找向量，并返回各行及矩阵 Shape；`dotProductSteps` 返回向量点积的逐项乘积与累计和；`multiplyMatrices` 返回显式 Shape 的矩阵乘法结果，`matrixCellSteps` 拆解一个输出格的行乘列点积；`projectQKV` 用固定 `Wq`、`Wk`、`Wv` 生成三个投影矩阵。这些学习模块都不读取网页，也不访问网络。

`src/projection-view.js` 单独负责本阶段矩阵与输出格的 DOM 显示；`src/app.js` 只把 Analyze 得到的 Embedding 快照交给它。其余文件负责让实验可以操作：`index.html` 提供页面结构，`src/styles.css` 设置页面布局，`server.mjs` 提供本地静态服务，`tests/` 中的测试固定学习规则以便回归检查。页面与样式代码是工程支撑，不必深入研究。

## 学习路线图

当前已实现 Phase 1、Phase 2、Phase 3 和 Phase 4。后续阶段计划如下：

1. **Phase 5 — Attention Score：** 计算 `QKᵀ`，理解为什么 N 个 Token 会产生 `N × N` 矩阵。
2. **Phase 6 — Scaled Dot-Product Attention：** 分步展示缩放、Softmax、权重和加权求和。
3. **Phase 7 — Attention Heatmap：** 按 Query 和 Key 位置探索实际权重。
4. **Phase 8 — Multi-Head Attention：** 比较不同 Head，查看拼接与输出投影。
5. **Phase 9 — Transformer Block：** 跟踪 Attention、残差连接、归一化和前馈网络。
6. **Phase 10 — 位置信息：** 比较不同顺序的序列，观察加入位置信息后的变化。
7. **Phase 11 — Mask：** 可视化 Padding Mask 与 Causal Mask。
8. **Phase 12 — Transformer 架构：** 串起 Encoder 和 Decoder 模块及其 Shape。
9. **Phase 13 — TensorFlow：** 将框架实现与此前手动拆解的运算对照。
10. **Phase 14 — Tiny Transformer：** 观察一个小模型的预测、Loss、Gradient 和权重更新。

每个新概念优先按“直觉 → 数学 → Shape → 代码”的顺序学习。前期会使用少量 Token 和小矩阵，使中间数值容易检查。框架语法是后续对照已有概念的工具，不会变成独立的 JavaScript 或 Python 入门课程。
