import sys
import gc
PLACE_HOLDER = '_'
#"""

def read(file):
    S = []
    for i in range(len(file)):
        elements = file[i].split(' ')
        s = []
        for e in elements:
            s.append(e.split())
        S.append(s)
    return S

"""
def read(filename):
    S = []
    with open(filename, 'r') as input:
        for line in input.readlines():
            elements = line.split(',')      #分隔符
            s = []
            for e in elements:
                s.append(e.split())
            S.append(s)
    #print(S)
    return S
"""
class SquencePattern:
    def __init__(self, squence, support):
        self.squence = []
        for s in squence:
            self.squence.append(list(s))  #元素换成列表
        self.support = support
    def append(self, p):                           #如果第一位是‘_’则remove后再加到squence里面

        if p.squence[0][0] == PLACE_HOLDER:
            first_e = p.squence[0]
            first_e.remove(PLACE_HOLDER)  #将‘_’删掉
            self.squence[-1].extend(first_e)
            self.squence.extend(p.squence[1:])
        else:
            self.squence.extend(p.squence)
        self.support = min(self.support, p.support)

#threshold  支持度阈值
def prefixSpan(pattern, S, threshold):
    gc.disable()
    patterns = []
    f_list = frequent_items(S, pattern, threshold)
    for i in f_list:
        p = SquencePattern(pattern.squence, pattern.support)
        p.append(i)
        patterns.append(p)
        p_S = build_projected_database(S, p)
        p_patterns = prefixSpan(p, p_S, threshold)
        patterns.extend(p_patterns)
        del p_patterns
    gc.enable()
    return patterns

def frequent_items(S, pattern, threshold):
    items = {}
    _items = {}
    f_list = []
    if S is None or len(S) == 0:
        return []

    if len(pattern.squence) != 0:
        last_e = pattern.squence[-1]
    else:
        last_e = []
    for s in S:
        # class 1
        #print(s)
        is_prefix = True
        for item in last_e:
            if item not in s[0]:
                is_prefix = False
                break
        # print(is_prefix)
        if is_prefix and len(last_e) > 0:
            index = s[0].index(last_e[-1])
            if index < len(s[0]) - 1:
                for item in s[0][index + 1:]:
                    if item in _items:
                        print(s)
                        _items[item] += 1
                    else:
                        _items[item] = 1
        # class 2
        if PLACE_HOLDER in s[0]:
            for item in s[0][1:]:
                if item in _items:
                    _items[item] += 1
                else:
                    _items[item] = 1
            s = s[1:]
        # class 3
        counted = []
        for element in s:
            for item in element:
                if item not in counted:
                    counted.append(item)
                    if item in items:
                        items[item] += 1
                    else:
                        items[item] = 1
    f_list.extend([SquencePattern([[PLACE_HOLDER, k]], v) for k, v in _items.items() if v >= threshold])
    f_list.extend([SquencePattern([[k]], v) for k, v in items.items() if v >= threshold])
    sorted_list = sorted(f_list, key=lambda p: p.support)   #lambda是虚拟函数
    del f_list
    return sorted_list

def build_projected_database(S, pattern):
    """
    suppose S is projected database base on pattern's prefix,  pattern是当前的前缀，S为其投影数据库
    so we only need to use the last element in pattern to
    build projected database
    """
    # print(S)
    # print(pattern.squence)

    p_S = []
    last_e = pattern.squence[-1]
    last_item = last_e[-1]
    for s in S:
        p_s = []
        for element in s:
            is_prefix = False
            if PLACE_HOLDER in element:
                if last_item in element and len(pattern.squence[-1]) > 1:
                    is_prefix = True
            else:
                is_prefix = True
                for item in last_e:
                    if item not in element:
                        is_prefix = False
                        break
            if is_prefix:
                e_index = s.index(element)
                i_index = element.index(last_item)
                if i_index == len(element) - 1:
                    p_s = s[e_index + 1:]
                else:
                    p_s = s[e_index:]
                    # index = element.index(last_item)
                    e = element[i_index:]
                    e[0] = PLACE_HOLDER
                    p_s[0] = e
                break

        if len(p_s) != 0:
            p_S.append(p_s)
    return p_S


def print_patterns(patterns):
    for p in patterns:
        print("pattern:{0}, support:{1}".format(p.squence, p.support))

def get_maxPatterns(patterns):
   #找support最大的pattern
    max = 0
    for p in patterns:
        if(len(p.squence)>=2 and (['-1'] not in p.squence) and isLegal(p)==1):
            if(p.support >= max):
                max = p.support
    #重新遍历 查看是否有相同support的模式；
    patterns_list = []
    for p in patterns:
        if((p.support == max) and (len(p.squence)>=2) and (['-1'] not in p.squence) and isLegal(p)==1):
            patterns_list.append(p)
    return patterns_list,max

def isLegal(p):
    x = len(p.squence)
    for i in range(x-1):
        if(p.squence[i]!=p.squence[i+1]):
            return 1
    return 0

if __name__ == "__main__":

    S = read("1111.txt")
    patterns = prefixSpan(SquencePattern([], sys.maxsize), S, 2)
    #print_patterns(patterns)
    #patterns = get_maxPatterns(patterns)
    #print_patterns(patterns)
