import itertools
import pickle
from typing import Any, List

import pandas as pd


def generate_sorting_function(sched_alg: str):  # 根据调度算法返回排序函数
    if sched_alg == 'fifo':
        sort_func = lambda x: x.arrival if not x.preempt_cloud else x.new_arrival  # 没有抢占时返回arrival，抢占时返回抢占后的新到达时间
    elif sched_alg == 'lifo':
        sort_func = lambda x: -x.arrival if not x.preempt_cloud else x.new_arrival  # 取负值，降序排列
    elif sched_alg == 'edf':
        sort_func = lambda x: x.deadline  # 截止时间最早
    elif sched_alg == 'evdf':
        sort_func = lambda x: x.deadline * x.num_gpus  # 加权截止时间最早
    elif sched_alg == 'ldf':
        sort_func = lambda x: -x.deadline  # 最迟的截止时间优先
    elif sched_alg == 'sjf':
        sort_func = lambda x: x.runtime  # 最短任务优先
    elif sched_alg == 'svjf':
        sort_func = lambda x: x.cost  # 加权最短任务优先，最少成本优先
    elif sched_alg == 'ljf':
        sort_func = lambda x: -x.runtime
    elif sched_alg == 'lvjf':
        sort_func = lambda x: -x.cost
    elif sched_alg == 'swf':
        sort_func = lambda x: x.deadline - x.runtime  
    elif sched_alg == 'svwf':
        sort_func = lambda x: (x.deadline - x.runtime) * x.num_gpus
    elif sched_alg == 'lwf':
        sort_func = lambda x: -x.deadline + x.runtime
    else:
        raise ValueError(
            f'Scheudling algorithm {sched_alg} does not match existing policies.'
        )
    return sort_func


def convert_to_lists(d: dict):   # 输入是一个字典d，该函数遍历字典并将其中的值转换为列表
    for key, value in d.items():
        # If the value is a dictionary, recursively convert it to a list
        if isinstance(value, dict):
            d[key] = convert_to_lists(value)  # 如果值是字典则递归转换
        elif not isinstance(value, list):
            d[key] = [value]  # 如果值不是列表类型，则把该值转换为列表。
    return d
'''
例如（只转换了值）：
d = {
    'a': 1,
    'b': 'hello',
    'c': [1, 2, 3],
    'd': 5.6
}
则输出为：
{
    'a': [1],
    'b': ['hello'],
    'c': [1, 2, 3],
    'd': [5.6]
}
'''


def flatten_dict(nested_dict, parent_key='', sep=':', preserve_name=False):  # 将嵌套字典扁平化，使键变成单层键（通过 parent_key（连接键的前缀） 和 sep（连接键的分隔符，默认冒号） 连接）
    flattened_dict = {}
    for key, value in nested_dict.items():  
        if preserve_name:  # 是否保留原有的键名
            new_key = key
        else:  # 生成新的键名
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):  # 如果值是字典，则递归调用
            flattened_dict.update(
                flatten_dict(value,
                             new_key,
                             sep=sep,
                             preserve_name=preserve_name))
        else:
            flattened_dict[new_key] = value
    return flattened_dict


def unflatten_dict(flattened_dict, sep=':'):  # 扁平化的字典回复成嵌套字典
    unflattened_dict = {}
    for key, value in flattened_dict.items():
        parts = key.split(sep)
        current_dict = unflattened_dict
        for part in parts[:-1]:
            current_dict = current_dict.setdefault(part, {})
        current_dict[parts[-1]] = value
    return unflattened_dict


def generate_cartesian_product(d: dict):  # 生成字典值的笛卡尔积，返回每个组合的字典列表
    d = flatten_dict(d)
    print(d)
    # Get the keys and values from the outer dictionary
    keys = list(d.keys())
    values = list(d.values())

    # Use itertools.product to generate the cartesian product of the values
    product = itertools.product(*values)  # 生成值的笛卡尔积

    # Create a list of dictionaries with the key-value pairs for each combination
    result = [dict(zip(keys, p)) for p in product]  # 将每个笛卡尔积组合 p 转换为字典，并将所有字典保存在 result 列表中
    # Return the list of dictionaries
    return [unflatten_dict(r) for r in result]


def is_subset(list1: List[Any], list2: List[Any]):
    """Checks if list2 is a subset of list1 and returns the matching indexes of the subset.
    检查 list2 是否是 list1 的子集，并返回匹配的索引"""
    indexes = []
    for i2, elem in enumerate(list2):
        for i1, x in enumerate(list1):
            if x == elem and i1 not in indexes:
                indexes.append(i1)
                break
        if len(indexes) != i2 + 1:
            return []
    return indexes


def _load_logs(file_path: str):
    file = open(file_path, 'rb')  # 从 file_path 读取二进制数据
    return pickle.load(file)  # 使用 pickle 反序列化，返回数据


def load_logs_as_dataframe(file_path: str):
    simulator_results = _load_logs(file_path)
    for r in simulator_results:
        if 'snapshot' in r:
            r['snapshot'] = [r['snapshot']]  # 如果日志中包含snapshot字段，则转为列表
    simulator_results = [
        flatten_dict(r, preserve_name=True) for r in simulator_results  # 扁平化日志数据
    ]
    return pd.DataFrame(simulator_results)  # Pandas DataFrame 是 Python 中的一个二维表格数据结构，类似于 Excel 表格
