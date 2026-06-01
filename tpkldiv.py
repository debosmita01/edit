import numpy as np
import math
import os
import copy

def calc_tp_count(lvl, p_size, border_value=-1):
    padded_lvl = np.pad(lvl, p_size, constant_values=border_value)
    pattern_dict = {}
    for (y, x), v in np.ndenumerate(lvl):
        sy, sx = y+p_size-p_size//2, x+p_size-p_size//2
        pattern = str(padded_lvl[sy:sy+p_size,sx:sx+p_size])
        if not (pattern in pattern_dict):
            pattern_dict[pattern] = 0
        pattern_dict[pattern] += 1
    return pattern_dict

def calc_tp_kldiv(test_count, train_count, epsilon = 1e-6):
    p_dict = test_count
    q_dict = train_count
    t_patterns = set()
    t_patterns.update(p_dict.keys())
    t_patterns.update(q_dict.keys())
    total_p = sum(p_dict.values())
    total_q = sum(q_dict.values())
    #print(len(p_dict),q_dict)
    mp_dict = {}
    mq_dict = {}
    mp_total = 0
    mq_total = 0
    for x in t_patterns:
        p_dash = epsilon/((total_p + epsilon) * (1 + epsilon))
        q_dash = epsilon/((total_q + epsilon) * (1 + epsilon))
        if x in p_dict:
            p_dash = (p_dict[x] + epsilon) / ((total_p + epsilon) * (1 + epsilon))
        if x in q_dict:
            q_dash = (q_dict[x] + epsilon) / ((total_p + epsilon) * (1 + epsilon))
        mp_dict[x] = p_dash
        mq_dict[x] = q_dash
        mp_total += p_dash
        mq_total += q_dash

    value = 0
    for x in t_patterns:
        p_dash = mp_dict[x] / mp_total
        q_dash = mq_dict[x] / mq_total
        value += p_dash * math.log(p_dash/q_dash)
    return value

def build_indeces(lvls):
    normal_sames = ['M-']
    same = normal_sames
    full_string = ''
    for lvl in lvls:
        full_string += ''.join(lvl)
    uniqueSymbols = set(full_string)
    str2index = {}
    index2str = {}
    c_index = 0
    if same is not None:
        for g in same:
            for c in g:
                str2index[c] = c_index
            index2str[c_index] = g[0]
            c_index += 1
    for symbol in uniqueSymbols:
        if symbol not in str2index:
            str2index[symbol] = c_index
            index2str[c_index] = symbol
            c_index += 1
    return str2index, index2str, c_index

def get_integer_lvl(lvl, str2index):
    result = []
    numpyLvl = np.zeros((len(lvl), len(lvl[0])))
    for y in range(len(lvl)):
        for x in range(len(lvl[y])):
            c = lvl[y][x]
            numpyLvl[y][x] = str2index[c]
    return numpyLvl.astype(np.uint8)



def get_lvls(folder):
    lvls = []
    files = sorted([f for f in os.listdir(folder)])
    for f in files:
        if f.endswith(".txt"): 
            f=open(os.path.join(folder, f))
            lvl=f.readlines()
            clean_level = []
            for l in lvl:
                if len(l.strip()) > 0:
                    clean_level.append(l.strip())
            lvls.append(clean_level)
    return lvls

def get_level(filename):
    temp = open(filename).readlines()
    lvl = []    
    for l in temp:
        if(len(l.strip()) > 0):
            lvl.append(l.strip())
    return lvl


def get_str_lvl(level):
    lvl = []
    str_lvl = level.split("\n")
    for s in str_lvl:
        if len(s) > 0:
            lvl.append(s)
    return lvl

def get_tpkldiv(lvl, targets, window=3, w=0.5):
    score = []
    for t in targets:
        in_lvl = get_level(t)
        out_lvl = get_str_lvl(lvl)
        #print(in_lvl)
        #print(out_lvl)
        str2index, index2str, _ = build_indeces([in_lvl, out_lvl])
        in_lvl = get_integer_lvl(in_lvl, str2index)
        out_lvl = get_integer_lvl(out_lvl, str2index)
        in_count = calc_tp_count(in_lvl, window)
        out_count = calc_tp_count(out_lvl, window)
        value = w * calc_tp_kldiv(in_count, out_count) + (1-w) * calc_tp_kldiv(out_count, in_count)
        score.append(value)
    mean_score = np.mean(score)
    return mean_score

def get_min_tpkldiv(lvl, targets, window=3, w=0.5):
    score = []
    for t in targets:
        in_lvl = t
        out_lvl = get_str_lvl(lvl)
        #print(in_lvl)
        #print(out_lvl)
        str2index, index2str, _ = build_indeces([in_lvl, out_lvl])
        in_lvl = get_integer_lvl(in_lvl, str2index)
        out_lvl = get_integer_lvl(out_lvl, str2index)
        in_count = calc_tp_count(in_lvl, window)
        out_count = calc_tp_count(out_lvl, window)
        value = w * calc_tp_kldiv(in_count, out_count) + (1-w) * calc_tp_kldiv(out_count, in_count)
        score.append(value)
    min_score = np.min(score)
    return min_score

    
 