import numpy as np

f = 'WikiData.txt'
global NUM
NUM=0

# 加载数据
def load_data(f):
    # f：数据路径
    readtxt = np.loadtxt(open(f), dtype=int)
    edges = readtxt[:, :]
    # 从小到大排序，目的是让相同起点聚集在一起
    edges=sorted(edges, key=(lambda x: x[0]))
    global NUM
    NUM = np.max(edges)
    # 使下标与节点ID相同
    edges = list(map(lambda x:x-1,edges))
    # 返回值为排序后的数据
    return edges


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


def write_data(node_id, score):
    f = open('./result.txt', 'w')
    for i in range(100):
        f.write(str(node_id[i] + 1) + ' ' + str(score[node_id[i]]) + '\n')
    f.close()
    return


if __name__ == '__main__':
    data = load_data(f)
    dead_flag = dead_end(data)
    r = 100
    beta = input("please input β：")
    beta=float(beta)
    rank, re = block_stripe_pagerank(data, r, dead_flag, beta)
    re = re / np.sum(re)
    index = np.argsort(-re)
    write_data(index, re)
