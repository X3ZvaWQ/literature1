import math
import pandas
from paper_1_data import datas
from math import exp

# 定义常量
# 等宽字体起见，A=好 B=较好 C=一般 D=较差 E=差
STANDARD = ['F', 'E', 'D', 'C', 'B', 'A']
COMMENT = ['E', 'D', 'C', 'B', 'A']
# 标准评价模糊集的长度
G = 6
tau = 0.5

columns = ['C1', 'C2', 'C3', 'C4', 'C5']
index = ['S1', 'S2', 'S3', 'S4']

# 定义方法
def number_normalize_benefit(column):
    return column.map(lambda x: (x - column.min()) / (column.max() - column.min()))
def interval_normalize_benefit(column):
    # 区间数规范化 操作对象为一个DataFrames的一列 对应 (3.1)
    u_sum = column.map(lambda x: x[1]).sum()
    l_sum = column.map(lambda x: x[0]).sum()
    return column.map(lambda x: [x[0] / u_sum, x[1] / l_sum])

def calc_membership(interval, i, g):
    # 图3.1 给定[0，1]之间的区间数，需要计算的模糊集元素位置，标准语言评价集的长度，计算隶属度
    step = 1 / g
    if interval[0] >= (i + 1) * step:
        return 0
    if interval[1] <= (i - 1) * step:
        return 0
    if interval[0] <= i * step <= interval[1]:
        return 1
    if i * step < interval[0] < (i + 1) * step:
        def y(x): return (x - i * step) / step * (-1) + 1
        return y(interval[0])
    if (i-1) * step < interval[1] < i * step:
        def y(x): return (x - (i - 1) * step) / step 
        return y(interval[1])
def interval_to_fuzzyset(interval):
    set = []
    # 标准评价模型的隶属度计算
    for i in range(G + 1):
        set.append([i, calc_membership(interval, i, G)])
    return set
def order_to_interval(order, amount):
    # 评价顺序转换为区间数
    return [(amount - int(order)) / amount, ((amount - int(order) + 1)) / amount]

def calc_lang_membership(lang, i):
    # 这里俩尖尖函数计算起来太麻烦了，考虑到评价语言的长度是固定的，标准集也是固定的
    # 所以进行一个打表，打表直接用GeoGebra画，求交点了（
    membership_table = {
        'E': [1, 0.6, 0.2, 0, 0, 0, 0],
        'D': [0.4, 0.8, 0.8, 0.4, 0, 0, 0],
        'C': [0, 0.2, 0.6, 1, 0.6, 0.2, 0],
        'B': [0, 0, 0, 0.4, 0.8, 0.8, 0.4],
        'A': [0, 0, 0, 0, 0.2, 0.6, 1]
    }
    return membership_table[lang][i]
def lang_to_fuzzyset(column):
    # 将一列语言评价转换为模糊集
    def lang_to_fuzzyset_single(lang):
        # 将单个语言评价转换为模糊集
        fuzzy_set = []
        for i in range(G + 1):
            fuzzy_set.append([i, calc_lang_membership(lang, i)])
        return fuzzy_set
    return column.map(lang_to_fuzzyset_single)

def calc_lang_var_membership(lang_var, i):
    # 梯形+尖尖，更痛苦了，继续打表
    # 因为给的数据里边语言变量都是连续的，所以很方便
    key = lang_var[0] + lang_var[1]
    membership_table = {
        'ED': [1, 1, 0.8, 0.4, 0, 0, 0],
        'DC': [0.4, 0.8, 1, 1, 0.6, 0.2, 0],
        'CB': [0, 0.2, 0.6, 1, 1, 0.8, 0.4],
        'BA': [0, 0, 0, 0.4, 0.8, 1, 1]
    }
    return membership_table[key][i]
def lang_var_to_fuzzyset(column):
    # 将一列语言评价转换为模糊集
    def lang_var_to_fuzzyset_single(lang_var):
        # 将单个语言评价转换为模糊集
        fuzzy_set = []
        for i in range(G + 1):
            fuzzy_set.append([i, calc_lang_var_membership(lang_var, i)])
        return fuzzy_set
    return column.map(lang_var_to_fuzzyset_single)

def fuzzyset_to_number(fuzzyset):
    # 模糊集转换为单点值
    g = len(fuzzyset) - 1
    return sum([i[0] * i[1] for i in fuzzyset]) / (g * sum([i[1] for i in fuzzyset]))

# 进行操作
E = []  # 决策矩阵
# 依次将数据丢进决策矩阵
for d in datas:
    E.append(pandas.DataFrame(d, columns=columns, index=index))
# 对每个决策矩阵单独进行处理
for e in E:
    # 规范化处理
    e.loc[:, 'C1'] = number_normalize_benefit(e.loc[:, 'C1'])
    e.loc[:, 'C2'] = interval_normalize_benefit(e.loc[:, 'C2']).map(interval_to_fuzzyset).map(fuzzyset_to_number)
    e.loc[:, 'C3'] = e.loc[:, 'C3'].map(lambda x: order_to_interval(x, 4)).map(interval_to_fuzzyset).map(fuzzyset_to_number)
    e.loc[:, 'C4'] = lang_to_fuzzyset(e.loc[:, 'C4']).map(fuzzyset_to_number)
    e.loc[:, 'C5'] = lang_var_to_fuzzyset(e.loc[:, 'C5']).map(fuzzyset_to_number)

# 把所有决策矩阵加起来除以数量，获得集结矩阵
B = sum(E) / len(E)

W = {} # 权重
for j in columns:
    tmp = 0
    for i in index:
        for k in range(index.index(i)):
            _k = index[k]
            tmp += (B.loc[i, j] - B.loc[_k, j])
    W[j] = math.exp((tau/(1-tau)) * tmp - 1)
w_sum = sum(W.values())
for j in columns:
    W[j] /= w_sum
print(W)