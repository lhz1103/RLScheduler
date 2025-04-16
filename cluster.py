from node import Node
import utils

blocked_by_gpu_cpu_job = 0
blocked_by_cpu_job = 0


class Cluster(object):  # 实现了一个集群的类
    def __init__(self,
                 num_nodes,
                 num_gpus_per_node=8,
                 binpack='first-fit',  # 资源分配策略
                 backfill=False):
        self.num_nodes = num_nodes
        self.num_gpus_per_node = num_gpus_per_node
        # List of nodes in the cluster. Assumed homogeneity.
        self.nodes = [
            Node(num_gpus_per_node)
            for _ in range(num_nodes)
        ]  # 存储所有节点的列表
        # Maps Job ID to Job, active jobs running in the cluster
        self.active_jobs = {}  # 当前正在运行的任务
        # Maps Job ID to Job, reserved jobs to be scheduled in cluster
        self.reserved_jobs = {}  # 已预留但尚未运行的任务，尚不清楚预留是什么意思
        # This determines whether to binpack with backfill scheduling.
        self.backfill = backfill
        # Defines the bin packing algorithm, `first-fit`, `best-fit`.
        self.binpack = binpack

    def is_full(self):
        return all([n.free_gpus == 0 for n in self.nodes])  # 检查集群是否已满
        # all() 函数接收一个可迭代对象（这里是布尔值列表），当列表中的所有值都为 True 时，返回 True；否则返回 False
    def get_active_jobs(self):
        return self.active_jobs

    def try_fit_v2(self, cur_timestamp, job):  # 对一个任务进行资源分配
        global blocked_by_cpu_job
        global blocked_by_gpu_cpu_job
        num_gpus = job.num_gpus

        free_gpus = [n.free_gpus for n in self.nodes]
        # Quick check, no hope of fitting onto cluster :(
        if num_gpus > sum(free_gpus):
            return False, []

        # Generate job GPU demands
        if num_gpus > self.num_gpus_per_node:
            # Assume worst case colocation
            # Multinode case, i.e. 26 GPUs, 8 GPU/node cluster -> job_gpu_demands = [8,8,8,2]
            job_gpu_demands = [self.num_gpus_per_node] * int(  # 转成了一个列表，里面的元素就是每个节点上需求的GPU个数，[] * int就等于把[]里的元素复制int份，[8]*3=[8,8,8]
                num_gpus / self.num_gpus_per_node)  # 计算整除的节点需求个数
            if num_gpus % self.num_gpus_per_node:  # 如果GPU数量不是单节点容量的整数倍，则将剩余的GPU分配到一个额外节点
                job_gpu_demands.append(num_gpus % self.num_gpus_per_node)
        else:
            job_gpu_demands = [num_gpus]

        # =============================================================================
        # Generate Job Plans
        # =============================================================================
        # Go through free space only first, generate partial plan with free space
        # 初始化空闲GPU和cpu信息
        node_free_gpu_list = [  # 每个节点的空闲GPU索引列表
            list(range(self.num_gpus_per_node)) for _ in range(self.num_nodes)
        ]  # 对于每个节点，创建一个从 0 到 self.num_gpus_per_node - 1 的列表，表示该节点初始时所有 GPU 都是空闲的

        # Go through active jobs
        for a_job_idx, a_job in self.active_jobs.items():
            for n_idx, gpu_list in a_job.allocated_gpus.items():  # 遍历活跃任务分配的节点和GPU列表
                for gpu_idx in gpu_list:
                    node_free_gpu_list[n_idx].remove(gpu_idx)  # 空闲列表中移除分配的GPU索引，并减去对应节点空闲的GPU数量

        # Go through reserved jobs
        for r_job_idx, r_job in self.reserved_jobs.items():
            if r_job.start < cur_timestamp + job.runtime:  # 预留任务开始时间<现在时间+当前任务运行时间，则预留任务会影响新任务
                for n_idx, gpu_list in r_job.allocated_gpus.items():  # 遍历任务分配的节点和GPU列表，如果节点对应的GPU未被占用，则占用它，移除对应的GPU索引
                    for gpu_idx in gpu_list:  
                        if not self.nodes[n_idx].gpu_dict[gpu_idx]:  
                            node_free_gpu_list[n_idx].remove(gpu_idx)

        node_free_gpu_count = [len(g) for g in node_free_gpu_list]  # 对每个节点统计其空闲GPU数量

        node_free_count = [(i, node_free_gpu_count[i])  # 统计列表，每个元素为二元组（节点索引，节点空闲GPU数量）
                           for i in range(len(node_free_gpu_count))]
        if self.binpack == 'first-fit':  # 不排序，直接按初始顺序分配
            pass
        elif self.binpack == 'best-fit':  # 按节点空闲GPU数量递增排序，优先选择空闲 GPU 最少的节点以减小碎片化
            # Sort by nodes with the least free GPU(s).
            node_free_count.sort(key=lambda x: x[1])
        elif self.binpack == 'worst-fit':
            # Sort by nodes with the most free GPU(s). Don't use, very bad.
            node_free_count.sort(key=lambda x: x[1], reverse=True)
        elif self.binpack == 'tetris':
            # Sort nodes by the most free in terms of "normalized" dot product of free node resources and job resources (multi resource setting).
            pass
        else:
            raise ValueError(f'Invalid allocation strategy {self.binpack}!')

        # Maps node idx to list of gpu indexes for the job to take.
        temp = False
        node_idx_taken = {}  # 记录哪些节点被选中，存储每个节点为任务分配的GPU列表
        for list_idx, gpu_demand in enumerate(list(job_gpu_demands)):  # 遍历任务对GPU需求列表
            for n_idx, free_gpus in node_free_count:  # 遍历节点剩余资源
                if n_idx in node_idx_taken:  # 如果当前节点已经为这个任务分配过资源，则跳过
                    continue
                if free_gpus >= gpu_demand:
                    # 满足空闲的GPU数量大于等于需求
                    # TODO: Reserved GPUs in the beginning of list. Prioritize taking reserved.
                    node_idx_taken[n_idx] = node_free_gpu_list[
                        n_idx][:gpu_demand]  # 从 node_free_gpu_list 中选择当前节点前 gpu_demand 个 GPU 索引分配给作业，记录在 node_idx_taken
                    job_gpu_demands.remove(gpu_demand)  # 移除已满足的需求并break，开始处理下一个需求
                    break
                        # if job.num_gpus > 0 and not temp:
                        #     blocked_by_gpu_cpu_job += 1
                        #     temp = True

        # If there are still demands that cannot be satisifed via free and preempted jobs,
        # it cannot be scheduled on the cluster.
        if job_gpu_demands:  # 如果遍历完所有节点后，job_gpu_demands 仍然不为空，说明有未满足的 GPU 需求
            # if temp:
            #     print(
            #         f'GPU-CPU block occurrences: {blocked_by_gpu_cpu_job}, CPU block occurrences: {blocked_by_cpu_job}'
            #     )
            return False, []

        # =============================================================================
        # Execute Job Plans
        # =============================================================================
        # Job plan stores in `node_idx_taken`: {Node Index -> List of GPU Indexes}
        # node_idx_taken是之前生成的分配计划，形如 {节点索引: [GPU索引列表]}
        for n_idx, gpu_demand_list in node_idx_taken.items():
            node = self.nodes[n_idx]
            node.free_gpus -= len(gpu_demand_list)
            if node.free_gpus < 0:  # 检查有无分配超过可用资源
                raise ValueError('Ran out of cluster resources!')
            for idx in gpu_demand_list:
                if node.gpu_dict[idx] is not None:  # 检查节点的gpu_dict中对应的GPU索引是否被占用
                    raise ValueError('Generated execution plan is incorrect.')
                node.gpu_dict[idx] = job  # 将任务记录到节点的gpu_dict对应索引位置，表示已将GPU分配给该任务
            job.allocated_gpus[n_idx] = gpu_demand_list  # 该节点的GPU分配情况记录到任务的allocated_gpus中
        job.start = cur_timestamp
        self.active_jobs[job.idx] = job  # 记录任务开始时间并标记为活动状态

        return True, []

    def predict_wait(self, cur_timestamp, job, queue, loop=False):  # 预测新任务在给定的集群状态下是否可以等待
        max_timestamp = job.deadline - job.runtime  # 最晚开始时间是ddl-运行时间，ddl指等待时间的上限

        num_gpus = job.num_gpus

        def get_gpu_demand_list(cur_job):  # 辅助函数，用于根据任务需求计算所需GPU分布
            num_gpus = job.num_gpus
            if num_gpus > self.num_gpus_per_node:
                # Assume worst case colocation
                # Multinode case, i.e. 26 GPUs, 8 GPU/node cluster -> job_gpu_demands = [8,8,8,2]
                job_gpu_demands = [self.num_gpus_per_node] * int(
                    num_gpus / self.num_gpus_per_node)
                if num_gpus % self.num_gpus_per_node:
                    job_gpu_demands.append(num_gpus %
                                            self.num_gpus_per_node)
            else:
                job_gpu_demands = [num_gpus]
            
            return job_gpu_demands

        job_gpu_demands = get_gpu_demand_list(job)

        node_free_gpu_count = [0] * self.num_nodes  # 初始化节点可用GPU统计

        for n_idx, node in enumerate(self.nodes):
            for gpu_idx in range(self.num_gpus_per_node):
                if node.gpu_dict[gpu_idx]:
                    continue
                node_free_gpu_count[n_idx] += 1  # 统计每个节点空闲GPU数

        # "Plan" ahead of the queue
        # for q_job in queue:
        #     q_job_gpu_demands = get_gpu_demand_list(q_job)
        #     q_node_index = can_cluster_fit(node_free_gpu_count)

        active_job_list = [a_job for a_job in self.active_jobs.values()]  # 获取所有的活跃任务（正在运行的任务），按预计结束时间（开始+运行）升序排列
        active_job_list.sort(key=lambda x: x.start + x.runtime)

        def can_cluster_fit(free_gpu_count):  # 判断当前集群GPU能否满足任务需求
            node_indexes = []  # 记录满足GPU需求的节点索引
            for demand_idx, job_gpu_demand in enumerate(job_gpu_demands):
                for node_idx, free_gpus_node in enumerate(free_gpu_count):
                    if job_gpu_demand <= free_gpus_node \
                        and node_idx not in node_indexes:  # 这里的逻辑是节点不能共享，一旦某个节点被分配给了任务，即使它有空闲的GPU也不能分给其他任务
                        node_indexes.append(node_idx)
                    if len(node_indexes) == len(job_gpu_demands):  # 如果相等，表示每个需求都已经找到对应的节点可以满足
                        return node_indexes
            if len(node_indexes) != len(job_gpu_demands):
                return []
            return node_indexes

        if can_cluster_fit(node_free_gpu_count):
            return True # 表示可以等待

        for a_job in active_job_list:  # 这个不太明白
            if a_job.start + a_job.runtime > max_timestamp:
                return False
            for n_idx, gpu_list in a_job.allocated_gpus.items():
                node_free_gpu_count[n_idx] += len(gpu_list)

            if can_cluster_fit(node_free_gpu_count):
                return True
        return False

    # Backfill Scheduling: Reserve blocking job.
    def try_reserve(self, cur_timestamp, job):  # 尝试为当前任务预留GPU
        max_timestemp = job.deadline - job.runtime

        free_gpus = [n.free_gpus for n in self.nodes]
        active_job_list = [a_job for a_job in self.active_jobs.values()]
        active_job_list.sort(key=lambda x: x.start + x.runtime)

        num_gpus = job.num_gpus
        # Generate job GPU demands
        if num_gpus > self.num_gpus_per_node:
            # Multinode case, i.e. 26 GPUs, 8 GPU/node cluster -> job_gpu_demands = [8,8,8,2]
            job_gpu_demands = [self.num_gpus_per_node] * int(
                num_gpus / self.num_gpus_per_node)
            if num_gpus % self.num_gpus_per_node:
                job_gpu_demands.append(num_gpus % self.num_gpus_per_node)
        else:
            job_gpu_demands = [num_gpus]

        node_free_list = [[] for _ in range(self.num_nodes)]
        node_free_count = [0] * self.num_nodes
        for n_idx, node in enumerate(self.nodes):  # 构建节点的空闲资源列表
            for gpu_idx in range(self.num_gpus_per_node):
                if node.gpu_dict[gpu_idx] or node.reserved_gpus[gpu_idx]:
                    continue
                node_free_count[n_idx] += 1
                node_free_list[n_idx].append(gpu_idx)

        for a_job in active_job_list:
            if a_job.start + a_job.runtime > job.deadline - job.runtime:  # 活动任务的结束时间超过目标任务的最晚开始时间，就说明这个活动任务的资源无法预留
                return False
            for n_idx, gpu_list in a_job.allocated_gpus.items():
                for gpu_idx in gpu_list:
                    if self.nodes[n_idx].reserved_gpus[gpu_idx]:
                        continue
                    node_free_list[n_idx].append(gpu_idx)  # 否则可以作为潜在资源加到空闲资源列表里
                    node_free_count[n_idx] += 1

            node_indexes = utils.is_subset(node_free_count, job_gpu_demands)  # 判断job_gpu_demands是否是node_free_count的一个子集，并把二者相匹配的部分赋值给node_indexes
            if node_indexes:
                for idx, n_idx in enumerate(node_indexes):  # 为当前任务分配资源
                    gpu_list = node_free_list[n_idx][-job_gpu_demands[idx]:]
                    job.allocated_gpus[n_idx] = gpu_list
                    cur_node = self.nodes[n_idx]
                    for gpu_idx in gpu_list:
                        cur_node.reserved_gpus[gpu_idx] = job  # 把当前节点的保留GPU与相应任务建立联系
                self.reserved_jobs[job.idx] = job
                job.block_job_idx = a_job.idx  # 活跃任务就是阻塞任务
                job.start = a_job.start + a_job.runtime  # 活跃任务执行完后就是当前调度任务的开始时间
                return True
        raise ValueError('I should not go here!')

    def try_clear(self, t: float):  # 释放完成任务占用的资源并清理已完成的任务，同时激活符合条件的预留任务
        """Clears cluster of completed jobs at time t.
        """
        completed_jobs = []
        # Free jobs on the cluster which have completed.
        for job_idx, job in self.active_jobs.items():
            # If job has finished before time t...
            if t >= job.start + job.runtime:
                for node_idx, gpu_list in job.allocated_gpus.items():
                    cur_node = self.nodes[node_idx]
                    node_gpu_dict = cur_node.gpu_dict
                    for gpu_idx in gpu_list:
                        node_gpu_dict[gpu_idx] = None  # 释放GPU
                    cur_node.free_gpus += len(gpu_list)  # 更新空闲GPU与CPU数量
                completed_jobs.append(job)

        # Clears cluster of completed jobs.
        c_job_idx = []
        for job in completed_jobs:
            job.state = 'LOCAL'  # 标记为本地运行
            c_job_idx.append(job.idx)
            del self.active_jobs[job.idx]  # 删除任务

        # Go through reserved jobs
        r_job_delete_idx = []
        for r_job_idx, r_job in self.reserved_jobs.items():
            # Move reserved job to active jobs
            if r_job.block_job_idx in c_job_idx:  # 阻塞任务已完成
                if t > r_job.start:
                    raise ValueError('sus')
                for node_idx, gpu_list in r_job.allocated_gpus.items():
                    cur_node = self.nodes[node_idx]
                    for gpu_idx in gpu_list:  
                        cur_node.gpu_dict[gpu_idx] = r_job  # 为预留任务分配资源
                        cur_node.reserved_gpus[gpu_idx] = None  # 释放预留的标记
                    cur_node.free_gpus -= len(gpu_list)
                    if cur_node.free_gpus < 0 or cur_node.free_cpus < 0:
                        print(cur_node.free_gpus, cur_node.free_cpus)
                        import pdb
                        pdb.set_trace()
                        raise ValueError('Reserved job, insufficient space.')
                r_job_delete_idx.append(r_job_idx)
                self.active_jobs[r_job_idx] = r_job  # 把任务从预留列表中移除，加入活动列表

        for r_job_idx in r_job_delete_idx:
            del self.reserved_jobs[r_job_idx]

        return completed_jobs

    def __repr__(self):  # 打印当前集群状态
        repr_str = 'Cluster State:\n'
        for idx, n in enumerate(self.nodes):
            repr_str += f'Node {idx}: {n}\n'
        return repr_str
