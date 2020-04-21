import numpy as np

f = 'WikiData.txt'
outdata = './output/output'

block_size = 2000


# 加载数据
def load_data(f):
    readtxt = np.loadtxt(open(f), dtype=int)
    edges = readtxt[:, :]
    return edges


# 获取节点列表，然后把数据按照块大小进行分块
def block_data(edges, block_size):
    # edges为原始数据，block_size为块大小
    nodes = []
    count = 0
    at = 0
    dest = None
    for edge in edges:
        if edge[0] not in nodes:
            nodes.append(edge[0])
        if edge[1] not in nodes:
            nodes.append(edge[1])
        if count % block_size == 0:
            if dest:
                dest.close()
            dest = open(outdata + str(at) + '.txt', 'w')
            at += 1
        dest.write(edge)
        count += 1
    print(nodes)
    # 返回值nodes为节点列表，at为分解的数据块数
    return nodes, at



def write_data(node_id, score):
    f = open('./result.txt', 'w')
    for i in range(len(node_id)):
        f.write(str(node_id[i]) + ' ' + str(score[i]) + '\n')
    f.close()
    return


if __name__ == '__main__':
    data = load_data(f)
