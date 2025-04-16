# 集群中的服务器类
class Node(object):
    def __init__(self, num_gpus):
        self.num_gpus = num_gpus  # 可用的GPU总数
        # Maps gpu index to job occupying that gpu
        self.gpu_dict = {}  # 映射每个GPU索引到当前占用该GPU的任务
        # Maps gpu index to job reserving that gpu
        self.reserved_gpus = {}  # 映射每个 GPU 的索引到当前预留该 GPU 的任务
        for idx in range(self.num_gpus):
            self.gpu_dict[idx] = None
            self.reserved_gpus[idx] = None
        self.free_gpus = self.num_gpus  # 空闲GPU数量

    def __repr__(self):
        return f'GPU: {self.gpu_dict}'  # 展示 gpu_dict 的当前状态：每个 GPU 索引及其对应的任务（如果空闲则为 None）以及CPU的数量
