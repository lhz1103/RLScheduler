import numpy as np
import torch
from typing import List, Dict, Any
from gym import spaces
from gym.spaces import Sequence, Box
from job import Job
import copy
from job_generator import load_processed_jobs
from EMA_Normalizer import EmaNormalizer
from utils_order import _top_k_nodes_by_free

class JobSchedulingEnv():
    def __init__(self, num_nodes=8, num_gpus_per_node=8, cloud_cost_weight=1, max_jobs_per_ts=20, alpha=1.0, beta=2.0, gamma=3.0, kappa=0.5, invalid_penalty=5.0):
        """
        基于PyTorch的任务调度环境
        
        参数:
            total_jobs: 所有待调度的任务列表
            num_nodes: 服务器节点数量
            num_gpus_per_node: 每个节点的GPU数量
            cloud_cost_weight: 云成本的权重系数
        """
        super().__init__()
        # 动作空间定义 (一个优先级分数，用于排序任务)
        self.action_space = spaces.Box(
            low=0, high=1, 
            shape=(max_jobs_per_ts,),  # 每个任务一个优先级分数
            dtype=np.float32
        )
         # 环境参数
        self.num_nodes = num_nodes
        self.num_gpus_per_node = num_gpus_per_node
        self.cloud_cost_weight = cloud_cost_weight
        self.total_jobs = []

        self.max_jobs_per_ts = max_jobs_per_ts  # 每个时隙最大的调度任务数量（包括新到来的和队列中等待的）
        # 二维矩阵，gpu_left[n, k] = x，代表第n台服务器上的第k块GPU还有x小时释放，x=0则说明空闲
        self.gpu_left = np.zeros((self.num_nodes, self.num_gpus_per_node), dtype=np.float32)
        
        # 状态空间定义
        self.observation_space = spaces.Dict({
            # 可变长度的任务特征矩阵 (5维特征)
            'jobs': spaces.Box(
                low=-np.inf, 
                high=np.inf,
                shape=(max_jobs_per_ts, 5),  # 第一维动态变化改为固定的最大任务数量，这里5限制了智能体只能看到任务的哪些特征
                dtype=np.float32
            ),
            # 集群状态 (每个节点的可用GPU数)
            'cluster': spaces.Box(
                low=0,
                high=num_gpus_per_node,
                shape=(num_nodes,),
                dtype=np.int32
            ),
            # GPU占用情况
            'gpu_left': spaces.Box(
                low=0.0,
                high=np.finfo(np.float32).max,
                shape=(num_nodes, num_gpus_per_node),
                dtype=np.float32
            )
        })
        
       


        # 任务队列管理
        self.queue = []           # 等待调度的任务队列
        self.active_jobs = []     # 正在运行的任务
        self.finished_jobs = []   # 已完成的任务
        self.current_time = 0    # 当前模拟时间
        
        self.reward = 0.0         # 当前步的奖励值
        self.visible_jobs = []   # 当前时隙内可调度的任务
        self.wait_jobs = []     # 存储当前时隙动作为等待的任务
        
        # 统计数据
        self.process_time = []
        self.total_process_time = 0.0  # 累计处理时间
        self.wait_time = []
        self.total_wait_time = 0.0  # 累计等待时间
        self.jct = []
        self.cloud_cost = []
        self.total_cloud_cost = 0.0  # 累计云成本
        self._last_cloud_snapshot = 0.0  # 上一时隙时的累计云成本

        # 奖励系数
        self.alpha = alpha  # 任务量系数
        self.beta = beta  # 成本系数
        self.gamma = gamma  # 上云被拒绝系数
        self.kappa = kappa  # 任务在本地完成的奖励
        self.invalid_penalty = invalid_penalty  # 选到无效值的惩罚

        # 成本预算
        self.budget_cap = 0.0
        self.budget_remain = 0.0
        self.block_happend = False

        # 归一化器
        self.norm_W = EmaNormalizer(rho=0.97)
        self.norm_C = EmaNormalizer(rho=0.97)


        # 初始化环境状态
        #self.reset()
    
    def estimate_gpu_budget(jobs, N, G, eta=0.1):
        # ------------- 预处理 -------------
        jobs_sorted = sorted(jobs, key=lambda j: j.arrival)  # 按到达时隙
        gpu_free_now = [G] * N        # 当前每节点空闲 GPU
        gpu_release  = [[] for _ in range(N)]  # release[t] = [(time, g)], 时隙级别

        gpu_h_cloud = 0.0  # 

        # ------------- 离散事件模拟 -------------
        for j in jobs_sorted:
            t = j.arrival             # 当前时隙
            # ---- 更新释放队列 ----
            for n in range(N):
                gpu_release[n] = [(rt, g) for (rt, g) in gpu_release[n] if rt > t]
                gpu_free_now[n] = G - sum(g for (_, g) in gpu_release[n])

            need = j.num_gpus
            wait_slots = 0

            while True:
                # a) 当前空闲 GPU 是否够 ?
                total_free = sum(gpu_free_now)
                if total_free >= need:
                    break            # 可以在 wait_slots 时隙后启动

                # b) 下一个释放事件
                earliest = min(rt for node in gpu_release for (rt, _) in node)
                # 快进到 earliest
                for n in range(N):
                    released = [(rt, g) for (rt, g) in gpu_release[n] if rt == earliest]
                    gpu_free_now[n] += sum(g for (_, g) in released)
                    gpu_release[n] = [(rt, g) for (rt, g) in gpu_release[n] if rt > earliest]
                wait_slots = earliest - t

                if wait_slots * 20/60 > j.deadline:
                    break            # 等不到，必须上云

            if wait_slots * 20/60 > j.deadline:
                # ---- 上云，累计 GPU·h ----
                gpu_h_cloud += j.runtime * j.num_gpus
            else:
                # ---- 本地排程 ----
                # 简单策略：按节点剩余 GPU 降序分配
                need_left = need
                for n in np.argsort(gpu_free_now)[::-1]:
                    alloc = min(gpu_free_now[n], need_left)
                    if alloc > 0:
                        gpu_free_now[n] -= alloc
                        gpu_release[n].append((t + wait_slots + math.ceil(j.runtime*3), alloc))
                        need_left -= alloc
                    if need_left == 0:
                        break

        return (1+eta) * gpu_h_cloud


    def reset(self, seed = 2024) -> Dict[str, np.ndarray]:
        """
        重置环境到初始状态
        返回初始观察值
        """
        """
        # 使用Philly原始trace时，系统负载3.529
        """
        dataset_config = {  
        'dataset': 'philly_gen',
        'arrival_rate': 90,  # 每时隙30个任务
        'cv_factor': 1.0,
        'total_jobs': 1000,
        'seed': seed,
        'privacy_rate': 0.2,
        'job_runtime': 4.0
        }
        init_jobs = load_processed_jobs(dataset_config)
        # 集群初始状态：每个服务器的可用GPU数
        self.cluster_state = np.full(
            shape=(self.num_nodes,),
            fill_value=self.num_gpus_per_node,
            dtype=np.int32
        )

        self.gpu_left = np.zeros((self.num_nodes, self.num_gpus_per_node), dtype=np.float32)
        """
        在一开始就设置好deadline，如果是启发式算法的话则再修改
        把总体预算设置为所有任务都上云（最坏情况下）的0.6倍
        """
        budget = 0.0
        for job in init_jobs:
            if job.privacy:  # 隐私任务只能放在队列中
                job.deadline = 1e12
            else:    
                job.deadline = job.num_gpus * job.runtime * 1
                budget += job.cost
        
        self.budget_cap = budget * 0.004
        self.budget_remain = budget * 0.004
        self.block_happend = False

        # 归一化器
        self.norm_W = EmaNormalizer(rho=0.97)
        self.norm_C = EmaNormalizer(rho=0.97)

        # 任务队列管理
        self.total_jobs = copy.deepcopy(init_jobs)  # 总体的任务队列
        self.queue = []           # 等待调度的任务队列
        self.active_jobs = []     # 正在运行的任务
        self.finished_jobs = []   # 已完成的任务
        self.current_time = 0    # 当前模拟时间
        self.total_cloud_cost = 0.0  # 累计云成本
        self.reward = 0.0         # 当前步的奖励值
        self.visible_jobs = []   # 当前时隙内可调度的任务

        self.process_time = []
        self.total_process_time = 0.0  # 累计处理时间
        self.wait_time = []
        self.total_wait_time = 0.0  # 累计等待时间
        self.jct = []  # 任务总JCT
        self.cloud_cost = []
        self.total_cloud_cost = 0.0  # 累计云成本
        self._last_cloud_snapshot = 0.0
        
        # 加载初始任务到队列
        self._load_jobs()
        return self._get_observation()

    
    
    
    def check_waiting_status(self, job: Job):
        """步骤2：判断任务是否等待中，且是否超过等待时间阈值"""
        if job.waiting:
            # 计算等待的时间是否超过上限
            if job.waitingtime >= job.deadline:
                # 等待时间超过阈值，进入步骤 3：判断费用上限
                self.handle_cloud_allocation(job)
            else:
                # 任务还在等待，可以继续判断是否需要等待
                self.handle_waiting_or_return(job)
        else:
            # 如果任务不在等待队列，进入步骤4：判断是否继续等待
            self.handle_waiting_or_return(job)
            
    def handle_cloud_allocation(self, job: Job):
        """步骤3：判断是否有足够的预算上云"""
        if self.total_cloud_cost + job.cost <= self.budget_cap:
            # 云费用未超过上限，选择上云
            self.allocate_to_cloud(job)
        else:
            # 超过预算，放回等待队列
            self.wait_jobs.append(job)
            job.waiting = True
            job.waitingtime += 20 / 60
            if job.waitingtime < job.deadline:  # 还没到deadline，虽然在步骤4中判断已经没必要等待了，但是理论上也可以继续等下去
                self.visible_jobs.remove(job)
            else:  # 已经超时需要上云，但由于预算原因无法上云，此时触发block
                self.block_happend = True
                self.visible_jobs.remove(job)

    def handle_waiting_or_return(self, job: Job):
        """步骤4：根据预测的等待时间判断是否放回等待队列"""
        # 预测等待时间（模拟）
        predicted_wait_time = self.predict_waiting_time(job)
        
        if predicted_wait_time <= (job.deadline - job.waitingtime):
            # 预测时间小于等于阈值，可以继续等待
            self.wait_jobs.append(job)
            job.waiting = True
            job.waitingtime += 20 / 60  # 更新等待时间
            self.visible_jobs.remove(job)
        else:
            # 等待时间太长，尝试上云
            self.handle_cloud_allocation(job)
    
    def predict_waiting_time(self, job: Job):
        need = job.num_gpus - sum(job.assigned_gpus)
        if need <= self.num_gpus_per_node:
            return self._predict_wait_single(need)  # 单服务器任务
        else:
            return self._predict_wait_multi(need)  # 多服务器任务
    
    def _predict_wait_single(self, need):
        """
        针对单节点任务，预测集群凑出所需GPU的最小时间
        """
        best = np.inf
        for node in range(self.num_nodes):
            free_now = self.cluster_state[node]
            if free_now >= need:  # 本地现在就能满足的情况，理论上不会发生
                return 0.0
            # 还差 k 块 → 找该 node 上第 k 小释放时间
            k = need - free_now
            busy = self.gpu_left[node][self.gpu_left[node] > 0]
            if len(busy) >= k:
                t = np.partition(busy, k-1)[k-1]  # 快速找到第 k 小的元素，并保证它左边都是更小（或相等）的数，右边是更大的数，但左右两边内部不需要完全排序。
                best = min(best, t)
        return best
    
    def _predict_wait_multi(self, need):
        """
        针对多节点任务，预测集群凑出所需GPU的最小时间
        """
        if need <= np.sum(self.cluster_state):  # 现在就能满足，理论上不会发生
            return 0.0
        # 拉平全集群 busy GPU 剩余时间
        all_busy = self.gpu_left[self.gpu_left > 0]  # 取出所有在忙的GPU的剩余时间，并打平成一个一维数组
        free_now = np.sum(self.cluster_state)
        lack = need - free_now          # 还缺多少
        if len(all_busy) < lack:  
            return np.inf               # 本地彻底凑不齐（整个集群都凑不出），理论上应该不会发生
        t = np.partition(all_busy, lack-1)[lack-1]  # 找出第lack小的释放时间
        return t

    def allocate_to_cloud(self, job:Job):
        cloud_cost = job.cost
        self.cloud_cost.append(cloud_cost)
        self.total_cloud_cost += cloud_cost
        job.start = self.current_time
        job.state = 'CLOUD'

        # 记录处理的时间，在云端不考虑放置，因此使用runtime
        self.process_time.append(job.runtime)
        self.total_process_time += job.runtime

        # 记录等待时间
        self.wait_time.append(job.waitingtime)
        self.total_wait_time += job.waitingtime
        
        jct = job.waitingtime + job.runtime
        self.jct.append(jct)
        
        self.finished_jobs.append(job)
        self.visible_jobs.remove(job)

    def allocate_one_job(self, job:Job):
        required_gpus = job.num_gpus - sum(job.assigned_gpus)
        allocated = False
        """
        步骤1：检查本地资源
        """
        # 单服务器任务：required_gpus <= self.num_gpus_per_node
        if required_gpus <= self.num_gpus_per_node:
            # Best-Fit：挑空闲 GPU 最少且能装下的节点
            cand = [(free, idx) for idx, free in enumerate(self.cluster_state)
                     if free >= required_gpus]
            if cand:                                     # 找得到
                free_gpu, node = min(cand)               # 剩余最少者
                self.cluster_state[node] -= required_gpus
                job.assigned_gpus[node] += required_gpus
                self._start_job_execution(job)
                return
        else:  # 多服务器任务
            remaining_gpu = required_gpus
            used_nodes = {}

            # 选空闲最多的节点，逐个加直到满足需求
            cand_nodes = _top_k_nodes_by_free(self.cluster_state, self.num_nodes)
            for node in cand_nodes:
                if self.cluster_state[node] > 0:
                    used_gpus = min(self.cluster_state[node], remaining_gpu)
                    remaining_gpu -= used_gpus
                    used_nodes[node] = used_gpus
                    if remaining_gpu <= 0:
                        break
            
            if remaining_gpu <= 0:  # 资源已足够，开始执行
                for node, gpus in used_nodes.items():
                    job.assigned_gpus[node] += gpus
                    self.cluster_state[node] -= gpus
                self._start_job_execution(job)
                return
            
        #  集群资源无法满足时
        if job.privacy:
            self.wait_jobs.append(job)
            job.waiting = True
            job.waitingtime += 20 / 60
            self.visible_jobs.remove(job)
        else:
            self.check_waiting_status(job)

    def _calc_remaining_gpu_hours(self) -> float:
        """
        GPU·min of *all* unfinished jobs
        = (排队 + 等待 + 可见) 全量
            + 运行中作业的剩余量
        """
        rem = 0.0

        # ① 队列中尚未启动的任务
        for job in (self.queue + self.visible_jobs + self.wait_jobs):
            need = job.num_gpus - sum(job.assigned_gpus)
            rem += job.num_gpus * job.runtime          # 仍按原始运行时计入

        # ② 运行中的任务 —— 剩余 actual_time
        for job in self.active_jobs:
            rem += job.num_gpus * max(job.actual_time - job.processing_time, 0.0)

        return rem          # 单位：GPU·hours
    
    def _calc_delta_cloud_gpu_hours(self) -> float:
        """
        云费的“边际增加量”，只在时隙末尾调用一次
        """
        delta = self.total_cloud_cost - self._last_cloud_snapshot
        self._last_cloud_snapshot = self.total_cloud_cost
        return delta                      # GPU·min

    def step(self, action: int) -> tuple[Dict[str, np.ndarray], float, bool, dict]: 
        """
        执行一个动作
        
        参数:
            action: 智能体给出的整数 0…max_jobs_per_ts-1
                 - 若 idx 超出当前 visible_jobs 范围：给大惩罚，并调度 visible_jobs[0]
                 - 否则调度 visible_jobs[idx]
            
        返回:
            observation: 新的状态观察值
            reward: 获得的奖励
            done: 是否终止
            info: 附加信息        
        """
        
        # 加载新到达的任务到队列
        #self._load_jobs()

        reward = 0.0

        # 终止条件检查
        if self._is_done():
            return self._get_observation(), reward, True, {}
        
        
        """
        在训练中是while 循环，在没有任务时根本不会执行到step，这种情况下如何反馈奖励
        """
        # 处理空队列情况，推进时隙，同步更新运行任务的情况
        if not self.visible_jobs:
            if action != -1:
                reward -= self.invalid_penalty
                return self._get_observation(), reward, False, {}
            
            finished = self._update_active_jobs()   # 统计完成
            reward += self.kappa * finished

            W_raw = self._calc_remaining_gpu_hours()  # 剩余任务量
            C_raw = self._calc_delta_cloud_gpu_hours()  # 新增费用

            W = self.norm_W.update(W_raw)
            C = self.norm_C.update(C_raw)

            reward -= self.alpha * W + self.beta * C

            # 推进时隙，加载下一个时隙任务
            self.current_time += 1
            self._load_jobs()

            reward = np.clip(reward, -10, 10)
            return self._get_observation(), reward, False, {}
        
        
        # -------- 判断 idx 是否有效 --------
        if (action < 0) or (action >= len(self.visible_jobs) and len(self.visible_jobs) > 0):
            # 事件奖励：出现无效动作，大惩罚，并按默认顺序调度首个任务
            reward -= self.invalid_penalty
            current_job = self.visible_jobs[0]
        else:
            current_job = self.visible_jobs[action]

        self.allocate_one_job(current_job)
        if self.block_happend:
            reward -= self.gamma
            self.block_happend = False  # 惩罚一次后就恢复，不然会重复惩罚

        if not self.visible_jobs:  # 如果分配完任务visible_jobs为空，则说明刚好分配完了最后一个任务，这时需要计算时隙奖励
            finished = self._update_active_jobs()   # 统计完成
            reward += self.kappa * finished

            
            W_raw = self._calc_remaining_gpu_hours()  # 剩余任务量
            C_raw = self._calc_delta_cloud_gpu_hours()  # 新增费用

            W = self.norm_W.update(W_raw)
            C = self.norm_C.update(C_raw)

            reward -= self.alpha * W + self.beta * C

            # 推进时隙，加载下一个时隙任务，以及更新运行中的任务情况都必须在一起
            self.current_time += 1
            self._load_jobs()
        
        # 检查终止条件
        reward = np.clip(reward, -10, 10)
        done = self._is_done()
        return self._get_observation(), reward, done, {}

    def _load_jobs(self):
        """将到达当前时间的任务加载到队列"""
        for job in list(self.total_jobs):  # 遍历副本防止修改原列表
            if job.timeslot <= self.current_time:
                job.assigned_gpus = [0] * self.num_nodes
                self.queue.append(job)
                self.total_jobs.remove(job)
            else:
                break

        
        """
        把queue中的前max_jobs_per_ts个任务出队加入visible_jobs，如果当前队列queue中的任务数量<max_jobs_per_ts，以queue为准
        """
        # 计算实际出队任务数
        num_to_load = max(min(self.max_jobs_per_ts - len(self.visible_jobs), len(self.queue)), 0) 
        # 在添加任务到可见任务队列时，需要限制visible_jobs大小最大为max_jobs_per_ts，防止下一个时隙的任务加入后超出长度
        for _ in range(num_to_load):
            tmp_job = self.queue.pop(0)
            # 将前num_to_load个任务出队并赋值给visible_jobs
            self.visible_jobs.append(tmp_job)



    def _start_job_execution(self, job: Job):
        """本地启动任务执行"""
        job.start = self.current_time
        job.state = 'LOCAL'
        job.actual_time, used_nodes_minus_1 = self._get_actual_time(job)
        for node_id, g in enumerate(job.assigned_gpus):
            if g > 0:
                # 找该 node 上还空着的 gpu idx
                free_slots = np.where(self.gpu_left[node_id] == 0)[0][:g]
                # 填上 remaining time
                self.gpu_left[node_id, free_slots] = job.actual_time

        # 记录处理的时间（计算+通信延迟）
        self.process_time.append(job.actual_time)
        self.total_process_time += job.actual_time

        # 记录等待时间
        self.wait_time.append(job.waitingtime)
        self.total_wait_time += job.waitingtime

        jct = job.actual_time+job.waitingtime
        self.jct.append(jct)

        self.active_jobs.append(job)
        self.visible_jobs.remove(job)
        #self.reward += ((job.timeslot + 1) / (job.start + 1)) * 5   # 在启动执行时，给一个正奖励函数，越早开始执行奖励越大（这个5是随便设置的）
        #self.reward -= used_nodes_minus_1  # 奖励再减去使用的节点数

    def _get_actual_time(self, job: Job, inter_nodes_factor=0.1) -> float:
        """计算考虑跨节点惩罚的实际运行时间"""
        used_nodes = sum(gpu > 0 for gpu in job.assigned_gpus)
        if used_nodes == 1:
            return job.runtime, 0
        return job.runtime * (1 + inter_nodes_factor * (used_nodes - 1)), used_nodes - 1

    def _update_active_jobs(self):
        """时隙结束，更新运行中的任务状态，返回完成的任务数"""
        # 把 wait_jobs 的全部元素搬到 queue 前端
        if self.wait_jobs:
            self.queue[0:0] = self.wait_jobs     # 等价于 queue = wait_jobs + queue

            # 或者：for job in self.wait_jobs[::-1]: self.queue.insert(0, job)

            # 清空 wait_jobs
            self.wait_jobs.clear()

        remaining = []
        finished_cnt = 0  #记录本时隙完成任务个数

        self.gpu_left = np.maximum(0, self.gpu_left - 20 / 60)  # 更新GPU的占用矩阵

        for job in self.active_jobs:
            job.processing_time += 20 / 60  # 模拟时间推进20分钟

            if job.processing_time >= job.actual_time:  # 任务完成
                # 释放GPU资源
                for node_id, gpus in enumerate(job.assigned_gpus):
                    self.cluster_state[node_id] += gpus
                self.finished_jobs.append(job)
                finished_cnt += 1                 
            else:
                remaining.append(job)
        self.active_jobs = remaining
        return finished_cnt

    def _get_observation(self) -> Dict[str, Any]:
        """构建观察值字典"""
               
        job_features = []
        for job in self.visible_jobs:
            feat = [
                job.idx,
                job.arrival,
                float(job.privacy),
                job.runtime,
                job.num_gpus - sum(job.assigned_gpus)
            ]
            job_features.append(feat)
        
        padding = [[0.0]*5 for _ in range(self.max_jobs_per_ts - len(self.visible_jobs))]
        job_features += padding

        return {
            'jobs': np.array(job_features, dtype=np.float32),  # 为啥一定要np.array？？
            'cluster': self.cluster_state.copy(),
            'gpu_left': self.gpu_left.copy()
        }

    def _is_done(self) -> bool:
        """判断是否终止"""
        done = not self.total_jobs and not self.queue and not self.active_jobs

        if done:
            print(f"\n=== Episode Completed! Finished jobs: {len(self.finished_jobs)} ===")

        return done
    
    
    def get_time_statistics(self):
        jct_mean = np.mean(self.jct)
        jct_p95 = np.percentile(self.jct, 95)
        jct_p99 = np.percentile(self.jct, 99)
        wait_time_mean = np.mean(self.wait_time)
        wait_time_p95 = np.percentile(self.wait_time, 95)
        wait_time_p99 = np.percentile(self.wait_time, 99)

        return jct_mean, jct_p95, jct_p99, wait_time_mean, wait_time_p95, wait_time_p99
    
    def get_cost_statistics(self):
        if len(self.cloud_cost) > 0:  
            cloud_cost_mean = np.mean(self.cloud_cost)
            cloud_cost_p95 = np.percentile(self.cloud_cost, 95)
            cloud_cost_p99 = np.percentile(self.cloud_cost, 99)
        else:
            # 如果 cloud_cost 为空，说明没有任务上传到云端，费用为0
            cloud_cost_mean = cloud_cost_p95 = cloud_cost_p99 = 0
        
        return cloud_cost_mean, cloud_cost_p95, cloud_cost_p99, self.total_cloud_cost, self.budget_cap

    def render(self, mode='human'):
        """可选：实现环境可视化"""
        print(f"Time: {self.current_time}")
        print(f"Cluster State: {self.cluster_state}")
        print(f"Queue Length: {len(self.queue)}")
        print(f"Visible Jobs: {len(self.visible_jobs)}")
        print(f"Active Jobs: {len(self.active_jobs)}")
        print(f"Finished Jobs: {len(self.finished_jobs)}")