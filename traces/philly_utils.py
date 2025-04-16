import datetime
import numpy as np

DATE_FORMAT_STR = '%Y-%m-%d %H:%M:%S'
MINUTES_PER_DAY = (24 * 60)
MICROSECONDS_PER_MINUTE = (60 * 1000)


def parse_date(date_str):  # 将一个日期字符串转换为 datetime 对象
    """Parses a date string and returns a datetime object if possible.

       Args:
           date_str: A string representing a date.

       Returns:
           A datetime object if the input string could be successfully
           parsed, None otherwise.
    """
    if date_str is None or date_str == '' or date_str == 'None':
        return None
    return datetime.datetime.strptime(date_str, DATE_FORMAT_STR)


def timedelta_to_minutes(timedelta):  # 将 timedelta 对象转换为以分钟为单位的数值
    """Converts a datetime timedelta object to minutes.

       Args:
           timedelta: The timedelta to convert.

       Returns:
           The number of minutes captured in the timedelta.
    """
    minutes = 0.0
    minutes += timedelta.days * MINUTES_PER_DAY  # 时间差中的天数，乘以每一天的分钟数，得到天数部分对应的分钟数
    minutes += timedelta.seconds / 60.0  # 时间差中的秒数，除以 60 后得到分钟数
    minutes += timedelta.microseconds / MICROSECONDS_PER_MINUTE  # 时间差中的微秒数，除以每分钟的微秒数，得到微秒部分对应的分钟数
    return minutes  # 将这三部分相加，得到总的分钟数


def get_cdf(data):  # 计算输入数据的累积分布函数
    """Returns the CDF of the given data.

       Args:
           data: A list of numerical values.

       Returns:
           An pair of lists (x, y) for plotting the CDF.
    """
    sorted_data = sorted(data)  # 从小到大进行排序
    p = 100. * np.arange(len(sorted_data)) / (len(sorted_data) - 1) 
    """
    np.arange(len(sorted_data)) 生成一个从 0 到 len(sorted_data) - 1 的整数数组，表示每个数据点在排序后的索引;
    100. * np.arange(...) / (len(sorted_data) - 1) 将这些索引转换为百分比，表示每个数据点对应的累积百分比
    """
    return sorted_data, p
