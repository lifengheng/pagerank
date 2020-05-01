# pagerank（.exe在dist下）

## 算法背景

​         Pagerank算法，即网页排名，又或Google排名，是Google创始人Sergey Brin和Lawrence Page在1997年构建早期的搜索系统原型提出的链式分析算法。根据网页被引用的次数以及引用该网页的网页的重要性来决定其重要性。

![](images/PageRank.png)

## 算法实现

pagerank的核心算法基于以下两种假设

1. 数量假设：如果有很多页面指向同一页面，那么这一页面就很重要。

2. 质量假设：如果一高质量页面指向某一页面，则该页面也可能很重要。

   

我们定义每一个页面的排名与指向该页面的数量和质量有关，但是每个页面都可能指向多个页面，就需要考虑该页面的出度。对于任一个页面j，可以求得它的排名$r_j$，其中$d_i$是页面i的出度
$$
r_j=\sum_{i\rightarrow j}\frac{r_i}{di}
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

先进行检查是否存在dead end

```
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



### 解决稀疏矩阵

对于互联网上的大多数网页，相互有联系的网页其实很少，因此采用矩阵存储十分浪费空间。解决稀疏矩阵的一个办法是用链表来存储，可以将$O(n^2)$的存储开销降为$O(kn)$，对于规模上亿的情况，提升的效果很可观。在python中

