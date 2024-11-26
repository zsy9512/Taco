'''
Author: shengyu
Date: 2024-05-09 19:33:40
LastEditors: shengyu
LastEditTime: 2024-05-16 08:49:17
Description: 算术编码器
'''

import os
import math

    
# 原始代码，通过计算文件中不同字节的频数和累积频数进行编码
def cal_pr(data):
    """
    pro_dic_pre:{2: 37, 230: 4, 169: 28, 106: 19, 140: 28, 207: 48, 178: 7, 242: 3, 59: 1}
    
    keys:[207, 2, 169, 140, 106, 178, 230, 242, 59]
    sym_pro:[48, 37, 28, 28, 19, 7, 4, 3, 1]
    accum_pro[0, 48, 85, 113, 141, 160, 167, 171, 174, 175]
    pro_dic_after:{2: [37, 48], 230: [4, 167], 169: [28, 85], 106: [19, 141], 140: [28, 113], 207: [48, 0], 178: [7, 160], 242: [3, 171], 59: [1, 174]}
    """
    pro_dic = {}
    data_set = set(data)
    for i in data_set:
        pro_dic[i] = data.count(i)  # 统计频数，一个字典 {keys:sym_pro}
    sym_pro = []  # 频数列表,每个字节对应的频数
    accum_pro = []  # 累积频数列表
    keys = []  # 字节名列表
    accum_p = 0
    data_size = len(data)
    for k in sorted(pro_dic, key=pro_dic.__getitem__, reverse=True):
        sym_pro.append(pro_dic[k])
        keys.append(k)
    for i in sym_pro:
        accum_pro.append(accum_p)
        accum_p += i
    accum_pro.append(data_size)#最后一个循环里没加上
    tmp = 0
    for k in sorted(pro_dic, key=pro_dic.__getitem__, reverse=True):
        pro_dic[k] = [pro_dic[k], accum_pro[tmp]]
        tmp += 1
    return pro_dic, keys, accum_pro


def dec2bin(x_up, x_down, L):
    #理论上是用来找最优二进制串，但是结果不对，原理暂时不明
    bins = ""
    while ((x_up != x_down) & (len(bins) < L)):
        
        x_up *= 2
        if x_up > x_down:
            bins += "1"
            x_up -= x_down
        elif x_up < x_down:
            bins += "0"
        else:
            bins += "1"
    return bins

# 编码
def encode(data, pro_dic, data_size):
    #C_up/C_down为区间左边界，A_up/A_down为区间长度
    C_up =0
    A_up = A_down = C_down = 1
    for i in range(len(data)):
        C_up = C_up * data_size + A_up * pro_dic[data[i]][1]
        C_down = C_down * data_size
        A_up *= pro_dic[data[i]][0]
        A_down *= data_size
    L = math.ceil(len(data) * math.log2(data_size) - math.log2(A_up))  # 计算编码长度
    #print(len(data) * math.log2(data_size),math.log2(A_up))
    #C_down=1686800000
    #bin_C = dec2bin(C_up, C_down, L)
    #理论上可以转化为概率，再进进行计算。但是有进位问题可能不准确，故弃用此函数
    bin_C = dec2Bin(Decimal(str(C_up/C_down)), Decimal(str((C_up/C_down)+(A_up/A_down))))
    amcode = bin_C[0:L]  # 生成编码
    #print(amcode)
    return C_up, C_down, amcode
#根据概率空间进行编码
from decimal import Decimal
# 计算数据中不同字节的概率
def cal_prob(data):
    prob_dic = {}
    data_set = set(data)
    for i in data_set:
        prob_dic[i] = data.count(i)  # 统计频数，一个字典 {keys:sym_pro}
    sym_pro = []  # 频数列表,每个字节对应的频数
    accum_pro = []  # 累积频数列表
    keys = []  # 字节名列表
    accum_p = 0
    data_size = len(data)
    
    for k in sorted(prob_dic, key=prob_dic.__getitem__, reverse=True):
        sym_pro.append(prob_dic[k])
        keys.append(k)
    for i in sym_pro:
        accum_pro.append(accum_p)
        accum_p += i
    accum_pro.append(data_size)#最后一个循环里没加上
    
    tmp = 0
    for k in sorted(prob_dic, key=prob_dic.__getitem__, reverse=True):
        prob_dic[k] = [Decimal(str(accum_pro[tmp]/data_size)) , Decimal(str(accum_pro[tmp+1]/data_size))]
        tmp += 1
    return prob_dic
def arithEncode(string, probDict):
    lower_bound = Decimal('0.0')
    upper_bound = Decimal('1.0')   
    for chr in string:
        intervalLength = upper_bound - lower_bound  # 区间长度
        # 不断更新区间上下界，注意必须先更新上界，否则会导致上界更新错误（因为上界的计算用的是上一次的下界）
        upper_bound = lower_bound + intervalLength * probDict[chr][1]
        lower_bound = lower_bound + intervalLength * probDict[chr][0]
    binString = dec2Bin(lower_bound, upper_bound)
    return binString
 # 求出最终区间内的最短二进制串
def dec2Bin(lower_bound, upper_bound):
    binStr = ''
    # 初始化01串转为的二进制数为0.0
    temp = Decimal(0.0)  
    i = 1  # 初始化幂为1
    while(1):
        bit = 1
        # 若当前位置1得到的数大于区间上界，因此实际应该将该位置0，此时temp不变
        if((temp + Decimal(1 / (2 ** i) * bit)) >upper_bound):
            bit = 0
            binStr += '0'
        # 若当前位置1得到的数恰好在区间内（即恰好大于下界），则该位置1不变，然后直接出循环，无需生成后续的位数
        else:
            if((temp + Decimal(1 / (2 ** i) * bit)) >= lower_bound):
                binStr += '1'
                break
            # 若当前位置1得到的数小于下界，表示后面还有余数，则该位置1,继续循环
            else:
                binStr += '1'
        # 更新temp
        temp += Decimal(1 / (2 ** i) * bit)
        # 阶数自增1，下一轮循环时用
        i += 1
    return binStr
#dic=cal_prob("AABABCABAB")
#arithEncode("AABABCABAB",dic)
# arithEncode("CABAB",dic)
#a,_,_=cal_pr("AABABCABAB")
#encode("AABABCABAB",a,10)
#encode("BCABAB",a,10)
# print("概率法编码结果：")
# data= b'\x8c\xa9\xcf\x02\xe6\xa9\xa9\xa9\xa9\xcf\x02j\xe6\xa9\xcf\xb2\x8c\x02\x8c\x02\xcf\xcf\x02\xa9\xe6\x02\x02\x02\x8c\xa9\x02\x02\xcf\xa9\x02\xa9j\x02\xcf\x02\x8c\xa9\xcf\x02\x8c\xe6;\xb2\xf2\x02\xf2\xcf\xb2\x8c\x02j\x02\xf2\x02\xa9\xcf\xcf\xcf\xa9\xcf\xcf\x02\xa9j\xb2\x8c\x02j\xa9\xcf\x02\xa9\x02j\xcf\xcf\x8c\x8cj\xa9\xcf\xb2\x8c\xcf\x02\x8c\x02\x02j\xcf\xcf\x02j\xcf\x02\xcf\x8c\xa9\x8c\x8c\xcf\xcf\xa9\xa9j\xcf\x02j\xcf\xcf\x02\x8c\x8c\xb2\x8cj\xa9\xcf\xcf\x02j\xcf\xa9\xcf\xa9\xa9\xa9j\xcf\xa9\xcf\xb2\x8c\x02\x02\x8c\x8c\x02\xcfj\x02\x8c\xcf\x02\x8c\xcf\x02\x8cj\xa9\x8cjj\x8c\x02\xa9j\xcf\xcf\xcf\xcf\x8c\xcf\xcf\xcf\xcf\x8c\xcf\xcf\xcf'
# dic=cal_prob(data)
# arithEncode(data,dic)
# print("原代码法编码结果：")
# dic,_,_=cal_pr(data)
# encode(data,dic,len(data))