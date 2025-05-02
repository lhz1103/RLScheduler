import numpy as np
import torch
from typing import List, Dict, Any
from gym import spaces
from gym.spaces import Sequence, Box
from job import Job
import copy
from job_generator import load_processed_jobs
from EMA_Normalizer import EmaNormalizer

class JobSchedulingEnv():
    def __init__(self, num_nodes=8, num_gpus_per_node=8, cloud_cost_weight=1, max_jobs_per_ts = 20, alpha=1.0, beta=2.0, gamma=3.0, kappa=0.5):
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

        self.max_jobs_per_ts = max_jobs_per_ts  # 每个时隙最大的调度任务数量（包括新到来的和队列中等待的）
        
        # 状态空间定义
        self.observation_space = spaces.Dict({
            # 可变长度的任务特征矩阵 (5维特征)
            'jobs': spaces.Box(
                low=-np.inf, 
                high=np.inf,
                shape=(max_jobs_per_ts, 5),  # 第一维动态变化改为固定的最大任务数量
                dtype=np.float32
            ),
            # 集群状态 (每个节点的可用GPU数)
            'cluster': spaces.Box(
                low=0,
                high=num_gpus_per_node,
                shape=(num_nodes,),
                dtype=np.int32
            )
        })
        
        # 环境参数
        self.num_nodes = num_nodes
        self.num_gpus_per_node = num_gpus_per_node
        self.cloud_cost_weight = cloud_cost_weight
        self.total_jobs = []


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
        self.cloud_cost = []
        self.total_cloud_cost = 0.0  # 累计云成本

        # 奖励系数
        self.alpha = alpha  # 任务量系数
        self.beta = beta  # 成本系数
        self.gamma = gamma  # 上云被拒绝系数
        self.kappa = kappa  # 任务在本地完成的奖励

        # 成本预算
        self.budget_cap = 0.0
        self.budget_left = 0.0
        self.block_happend = False

        # 归一化器
        self.norm_W = EmaNormalizer(rho=0.97, warmup_steps=15)
        self.norm_C = EmaNormalizer(rho=0.97, warmup_steps=15)


        # 初始化环境状态
        #self.reset()

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
        """
        在一开始就设置好deadline，如果是启发式算法的话则再修改
        把总体预算设置为所有任务都上云（最坏情况下）的0.6倍
        """
        for job in init_jobs:
            if job.privacy:  # 隐私任务只能放在队列中
                job.deadline = 1e12
            else:    
                job.deadline = job.cost * 1
                budget += job.cost
        
        self.budget_cap = budget * 0.6
        self.budget_left = budget * 0.6
        self.block_happend = False

        # 归一化器
        self.norm_W = EmaNormalizer(rho=0.97, warmup_steps=15)
        self.norm_C = EmaNormalizer(rho=0.97, warmup_steps=15)

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
        self.cloud_cost = []
        self.total_cloud_cost = 0.0  # 累计云成本
        
        # 加载初始任务到队列
        #self._load_jobs()
        return self._get_observation()
        
    def allocate_one_job(self, job:Job):
        # 尝试在集群中分配资源
        required_gpus = job.num_gpus - sum(job.assigned_gpus)
        allocated = False

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
            need_nodes = int(np.ceil(required_gpus / self.num_gpus_per_node))
        
        # 集群中有足够资源时
        if sum(self.cluster_state) >= required_gpus:
            # 策略1: 优先寻找能一次性分配的单节点
            for node_id in range(self.num_nodes):
                if self.cluster_state[node_id] >= required_gpus:
                    self.cluster_state[node_id] -= required_gpus
                    job.assigned_gpus[node_id] = required_gpus
                    self._start_job_execution(job)
                    allocated = True
                    break
            # 策略2: 跨节点分配
            if (not allocated) and (required_gpus <= sum(self.cluster_state)):
            # 跨节点分配逻辑（需计算实际运行时间）
                allocated_gpus = 0
                used_nodes = []

                for node_id in range(self.num_nodes):
                    alloc = min(self.cluster_state[node_id], required_gpus - sum(job.assigned_gpus))
                    if alloc > 0:
                        self.cluster_state[node_id] -= alloc
                        job.assigned_gpus[node_id] += alloc
                        allocated_gpus += alloc
                        used_nodes.append(node_id)
                    elif allocated_gpus >= required_gpus:
                        self._start_job_execution(job)
                        allocated = True
                        break   
        # 集群没有足够资源时 
        else:
            if job.deadline > 0:
                # 策略3: 等待
                self.wait_jobs.append(job)
                job.deadline -= 20 / 60
                job.waitingtime += 20 / 60
                self.visible_jobs.remove(job)
                # 区分是资源不足导致的等待（给小惩罚）还是有资源时主动等待（给一个大的惩罚）
                if all(x == 0 for x in self.cluster_state) or (sum(self.cluster_state) < required_gpus): # 如果是资源不足导致任务等待，给一个小惩罚
                    self.reward -= 5
                else:
                    self.reward -= 500
            # 策略4: 若无法在集群分配且允许上云
            elif not allocated and not job.privacy and job.deadline <= 0:
                cloud_cost = job.cost
                self.cloud_cost.append(cloud_cost)
                self.total_cloud_cost += cloud_cost
                job.start = self.current_time
                job.state = 'CLOUD'
                self.reward -= self.cloud_cost_weight * cloud_cost
                self.finished_jobs.append(job)
                self.visible_jobs.remove(job)
                allocated = True


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

        # 终止条件检查
        if self._is_done():
            return self._get_observation(), 0.0, True, {}
        
        # 处理空队列情况
        if not self.visible_jobs:
            #self.current_time += 1
            self._update_active_jobs()
            return self._get_observation(), self.reward, False, {}
        
        # -------- 判断 idx 是否有效 --------
        if (action < 0) or (action >= len(self.visible_jobs)):
            # 无效动作：大惩罚、按默认顺序调度首个任务
            self.reward -= 50.0
            current_job = self.visible_jobs[0]
        else:
            current_job = self.visible_jobs[action]

        self.allocate_one_job(current_job)
        
        # 检查终止条件
        done = self._is_done()
        return self._get_observation(), self.reward, done, {}

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
        """启动任务执行，并添加奖励"""
        job.start = self.current_time
        job.state = 'LOCAL'
        job.actual_time, used_nodes_minus_1 = self._get_actual_time(job)

        # 记录处理的时间（计算+通信延迟）
        self.process_time.append(job.actual_time)
        self.total_process_time += job.actual_time

        self.active_jobs.append(job)
        self.visible_jobs.remove(job)
        self.reward += ((job.timeslot + 1) / (job.start + 1)) * 5   # 在启动执行时，给一个正奖励函数，越早开始执行奖励越大（这个5是随便设置的）
        self.reward -= used_nodes_minus_1  # 奖励再减去使用的节点数

    def _get_actual_time(self, job: Job, inter_nodes_factor=0.1) -> float:
        """计算考虑跨节点惩罚的实际运行时间"""
        used_nodes = sum(gpu > 0 for gpu in job.assigned_gpus)
        if used_nodes == 1:
            return job.runtime, 0
        return job.runtime * (1 + inter_nodes_factor * (used_nodes - 1)), used_nodes - 1

    def _update_active_jobs(self):
        """更新运行中的任务状态"""
        # 把 wait_jobs 的全部元素搬到 queue 前端
        if self.wait_jobs:
            self.queue[0:0] = self.wait_jobs     # 等价于 queue = wait_jobs + queue

            # 或者：for job in self.wait_jobs[::-1]: self.queue.insert(0, job)

            # 清空 wait_jobs
            self.wait_jobs.clear()

        remaining = []
        for job in self.active_jobs:
            job.actual_time -= 20 / 60  # 模拟时间推进20分钟
            
            if job.actual_time <= 0:  # 任务完成
                # 释放GPU资源
                for node_id, gpus in enumerate(job.assigned_gpus):
                    self.cluster_state[node_id] += gpus
                self.finished_jobs.append(job)
                wait_time = (job.start - job.timeslot) * 20 / 60

                # 记录等待时间
                self.wait_time.append(wait_time)
                self.total_wait_time += wait_time
                
            else:
                remaining.append(job)
        self.active_jobs = remaining

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
            'cluster': self.cluster_state.copy()
        }

    def _is_done(self) -> bool:
        """判断是否终止"""
        done = not self.total_jobs and not self.queue and not self.active_jobs

        if done:
            print(f"\n=== Episode Completed! Finished jobs: {len(self.finished_jobs)} ===")

        return done
    
    
    def get_time_statistics(self):
        jct = self.process_time + self.wait_time
        jct_mean = np.mean(jct)
        jct_p95 = np.percentile(jct, 95)
        jct_p99 = np.percentile(jct, 99)
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
        
        return cloud_cost_mean, cloud_cost_p95, cloud_cost_p99

    def render(self, mode='human'):
        """可选：实现环境可视化"""
        print(f"Time: {self.current_time}")
        print(f"Cluster State: {self.cluster_state}")
        print(f"Queue Length: {len(self.queue)}")
        print(f"Visible Jobs: {len(self.visible_jobs)}")
        print(f"Active Jobs: {len(self.active_jobs)}")
        print(f"Finished Jobs: {len(self.finished_jobs)}")