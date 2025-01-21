import logging
import numpy as np
import math
import torch.nn.functional as F

def get_logger():
    """Get logging."""
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def next_batch(X1, X2, X3, X4, batch_size):
    """Return data for next batch"""
    tot = X1.shape[0]
    total = math.ceil(tot / batch_size)
    for i in range(int(total)):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        end_idx = min(tot, end_idx)
        batch_x1 = X1[start_idx: end_idx, ...]
        batch_x2 = X2[start_idx: end_idx, ...]
        batch_x3 = X3[start_idx: end_idx, ...]
        batch_x4 = X4[start_idx: end_idx, ...]
        # batch_x5 = X5[start_idx: end_idx, ...]

        yield (batch_x1, batch_x2, batch_x3, batch_x4, (i + 1))

def next_batch1(X1, X2, batch_size):
    """Return data for next batch"""
    tot = X1.shape[0]
    total = math.ceil(tot / batch_size)
    for i in range(int(total)):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        end_idx = min(tot, end_idx)
        batch_x1 = X1[start_idx: end_idx, ...]
        batch_x2 = X2[start_idx: end_idx, ...]
        # batch_x3 = X3[start_idx: end_idx, ...]
        # batch_x4 = X4[start_idx: end_idx, ...]

        yield (batch_x1, batch_x2, (i + 1))


# 求特征矩阵fea_mat1(nxd)和fea_mat2(nxd)的行向量之间的余弦相似度，得similarity为nxn的相似度矩阵, 元素取值范围为[-1,1]
def get_Similarity(fea_mat1, fea_mat2):
    a = fea_mat1.unsqueeze(1)
    b = fea_mat2.unsqueeze(0)
    Sim_mat = F.cosine_similarity(fea_mat1.unsqueeze(1), fea_mat2.unsqueeze(0), dim=-1)
    return Sim_mat

def cal_std(logger, missingrate, Lambda1, Lambda2, Lambda3,  *arg):
    """Return the average and its std"""
    if len(arg) == 3:
        print('ACC:'+ str(arg[0]))
        print('NMI:'+ str(arg[1]))
        print('ARI:'+ str(arg[2]))
        output = "ACC {:.3f} std {:.3f} NMI {:.3f} std {:.3f} ARI {:.3f} std {:.3f}".format( np.mean(arg[0]),
                                                                                             np.std(arg[0]),
                                                                                             np.mean(arg[1]),
                                                                                             np.std(arg[1]),
                                                                                             np.mean(arg[2]),
                                                                                             np.std(arg[2]))
    elif len(arg) == 1:
        print(arg)
        output = "ACC {:.3f} std {:.3f}".format(np.mean(arg), np.std(arg))

    print(output)

    # 指定文件路径
    file_path = "./data/scene-15.txt"

    # 生成数据，这里使用示例数据，你需要替换为你的实际数据
    data = 'missingrate:'+str(missingrate)+'-'+'l1:'+str(Lambda1)+'-'+'l2:'+str(Lambda2)+'-'+'l3:'+str(Lambda3)+'-----'+output+'\n'

    # 以追加模式打开文件，如果文件不存在会自动创建
    with open(file_path, "a") as file:
        file.write(data)

    return


def normalize(x):
    """Normalize"""
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x
