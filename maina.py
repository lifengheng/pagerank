import numpy as np

beta = 0.85
threshold = 1.0e-6
NUM = 8297


def fun(r):
    # 数据读入，为处理稀疏矩阵，将各条边以点对的形式储存在A中
    dt = np.dtype([('from', int), ('to', int)])
    A = np.empty((0), dtype=dt)
    for line in open("WikiData.txt"):
        data = line.partition("	")
        x = int(data[0])
        y = int(data[2])
        A = np.append(A, np.array([(x - 1, y - 1)], dtype=dt))
    # 将边的集合A以出发点的序号排序便于后续计算
    print(A)
    A = np.sort(A, order='from')
    print(A)
    # 利用w记录各点出度是否为零，便于发现dead end
    w = np.zeros((NUM))
    for a in A:
        w[a[0]] = 1
    v_new = np.ones(NUM) * 1 / NUM
    v_old = v_new
    B = v_new
    rank = 1
    # 当前后两次的各点的value值的距离和小于阈值threshold或迭代次数大于r时结束迭代
    while ((np.sum(np.abs(v_new - v_old)) > threshold or rank == 1) and rank < r):
        v_old = v_new
        x_old = A[0]
        start = end = 0
        v_new = np.zeros(NUM)
        for x in A:
            # 对于从同一点出发的边统一处理，仅需从A中读取从某一点出发的所有边即可，实现了分块矩阵
            if (x[0] != x_old[0]):
                for i in range(start, end):
                    v_new[A[i][1]] += v_old[x_old[0]] / (end - start)
                start = end
                x_old = x
                end += 1
            else:
                end += 1
        index = 0
        sum = 0
        # 对于dead end进行random teleport
        for x in w:
            if (x == 0):
                sum += v_old[index]
                index += 1
            else:
                index += 1
        v_new += sum * np.ones(NUM) * 1 / NUM
        v_new = beta * v_new + (1 - beta) * B
        rank += 1
    return rank, v_new


if __name__ == "__main__":
    rank, v = fun(100)
    v /= np.sum(v)
    w = np.argsort(-v)
    f = open("optimize_result.txt", "w")
    for i in range(100):
        f.writelines(str(w[i] + 1) + ' ' + str(v[w[i]]) + '\n')
