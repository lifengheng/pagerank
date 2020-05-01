# PageRank 算法实现与改进

## 算法背景

​         Pagerank算法，即网页排名，又或Google排名，是Google创始人Sergey Brin和Lawrence Page在1997年构建早期的搜索系统原型提出的链式分析算法。根据网页被引用的次数以及引用该网页的网页的重要性来决定其重要性。

![](images/PageRank.png)

## 数据分析

在`WikiData.txt`文档中，存储了连接的数据，数据有两列。表示了从 FromNodeID 到 ToNodeID 的连接。

## 实验原理

pagerank的核心算法基于以下两种假设

1. 数量假设：如果有很多页面指向同一页面，那么这一页面就很重要。

2. 质量假设：如果一高质量页面指向某一页面，则该页面也可能很重要。


我们定义每一个页面的排名与指向该页面的数量和质量有关，但是每个页面都可能指向多个页面，就需要考虑该页面的出度。对于任一个页面j，可以求得它的排名
$$
r_j
$$
，其中
$$
d_i
$$
是页面i的出度
$$
r_j=\sum_{i\rightarrow j}\frac{r_i}{d_i}
$$
化为矩阵运算可以得到
$$
\left |\begin{matrix}
r_1\\
r_2\\
\vdots\\
r_n
\end{matrix}
\right |  =
\left |
\begin{matrix}
M_{11} & M_{12} & \cdots & M_{1n}\\
M_{21} & M_{22} & \cdots & M_{2n}\\
\vdots&\vdots&\ddots&\vdots&\\
M_{n1} & M_{n2} & \cdots & M_{nn}\\
\end{matrix}
\right |*
\left |\begin{matrix}
r_1\\
r_2\\
\vdots\\
r_n
\end{matrix}
\right |
$$
其中
$$
M_{ij}=\begin{cases}
       \frac{i}{d_j}  \quad if \quad(j\rightarrow i)\\ 
       0    \quad others
\end{cases}
$$
记为
$$
R=MR
$$
### 解决spider trap

为了解决spider traps，引入了一个阻尼系数d，它表示人们在当前网页继续浏览的概率，而(1-d)就是随机打开一个网页的概率，这样整个图就是强连通的[^1]

[^1]: Brin S, Page L. The anatomy of a large-scale hypertextual web search engine[J]. 1998.




$$
r_j=d(\sum_{i\rightarrow j}\frac{r_i}{di})+(1-d)
$$
### 解决dead ends

dead ends会导致迭代过程中R的和不断减少，收敛不到正确结果。dead ends反映在矩阵M上就是某一列全为0，这会导致R所有元素都为0。因此在每次迭代都要加上失去的Rank值。也可以在对矩阵M进行预处理，将全0的列全部换为1/n。
$$
R_{new}=d(MR_{old})+[1-d*sum]_{n*1}
$$
这里的sum是$MR_{old}$各元素的求和。这样就保证了在迭代过程中R各元素之和始终为1。

### 优化稀疏矩阵

如果按照常规方式存储数据，大小为8297*8297，为了优化，我们选择按照原始数据那样存储，即from-to的形式

### 实现分块计算

由于在矩阵计算过程中，实际上，部分计算是重复的，因此，利用分块矩阵去修改它的值，会减少计算量，能够加快计算的效率。对于从同一点出发的边统一处理，仅需从A中读取从某 一点出发的所有边即可，实现了分块矩阵

## 实验步骤

### 引入包

我们只引入了numpy包来做矩阵运算处理

```python
import numpy as np
```

### 全局变量

f为数据的路径，`NUM`为点的数量，也就是最大的节点ID

```python
f = 'WikiData.txt'
global NUM
NUM=0
```

### 加载数据

调用`np.loadtxt()`方法来读取文件，并把数据按照从小到大排序，以便分块处理，然后让节点ID从0开始。

```python
# 加载数据
def load_data(f):
    # f：数据路径
    readtxt = np.loadtxt(open(f), dtype=int)
    edges = readtxt[:, :]
    # 从小到大排序，目的是让相同起点聚集在一起
    edges = sorted(edges, key=(lambda x: x[0]))
    global NUM
    NUM = np.max(edges)
    # 使下标与节点ID相同
    edges = list(map(lambda x: x - 1, edges))
    # 返回值为排序后的数据
    return edges
```

### 寻找deadend

`deadend`是没有出度的点，所以只需要遍历数据，看哪些点有就标记为1，没有就是0了。

```python
# 寻找dead_end
def dead_end(data):
    # data为所有数据
    global NUM
    dead_flag = np.zeros(NUM)
    # 若数据的起点没有该节点，则为0
    for d in data:
        dead_flag[d[0]] = 1
    # 返回值为一个01数组，0为dead_end
    return dead_flag
```

### 核心方法

该方法就是PageRank核心方法，

```python
# 核心函数
def block_stripe_pagerank(A, r, w, beta):
    # A：原始数据集，
    global NUM
    v_new = np.ones(NUM) * 1 / NUM  # 初始值
    v_old = v_new
    B = v_new
    rank = 1  # 迭代次数
    while 1:
        v_old = v_new  # 每次更新旧值
        x_old = A[0]  # 从第一个点开始
        start = end = 0  # 起点与终点
        v_new = np.zeros(NUM)
        for x in A:  # 遍历每条边
            # 对于从同一点出发的边统一处理,实现了分块矩阵
            if x[0] != x_old[0]:  # 遇到不同的起始点
                for i in range(start, end):  # 遍历相同起点的边
                    v_new[A[i][1]] += v_old[x_old[0]] / (end - start)  # 更新不同终点的入度
                start = end
                x_old = x  # 到下一个起始点
                end += 1
            else:
                end += 1  # 相同起点继续遍历
        index = 0
        sum = 0
        # 对于dead end进行random teleporting（心灵转移），就是我们认为在任何一个页面浏览的用户都有可能以一个极小的概率瞬间转移到另外一个随机页面
        for x in w:
            if (x == 0):  # 若是dead_end则把本来的值保留下来
                sum += v_old[index]
                index += 1
            else:
                index += 1
        # 加入到总的矩阵里面
        v_new += sum * np.ones(NUM) * 1 / NUM
        # 根据公式计算 beta为阻尼系数
        v_new = beta * v_new + (1 - beta) * B
        # 迭代次数加一
        rank += 1
        # 如果求的矩阵收敛或者迭代次数达到最大值则停止迭代
        if np.sum(np.abs(v_new - v_old)) < 1.0e-6 and rank < r:
            break;
    # 返回迭代次数，最后的矩阵
    return rank, v_new
```

