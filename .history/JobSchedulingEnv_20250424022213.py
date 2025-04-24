import numpy as np
import torch
from typing import List, Dict, Any
from gym import spaces
from gym.spaces import Sequence, Box
from job import Job
import copy
from job_generator import load_processed_jobs


class JobSchedulingEnv():
    def __init__(self, num_nodes=8, num_gpus_per_node=8, cloud_cost_weight=1, max_jobs_per_ts = 20):
        """
        基于PyTorch的任务调度环境
        
        参数:
            total_jobs: 所有待调度的任务列表
            num_nodes: 服务器节点数量
            num_gpus_per_node: 每个节点的GPU数量
            cloud_cost_weight: 云成本的权重系数
        """
        super().__init__()
        # 动作空间定义 (0~N-1: 服务器, N: 云, N+1: 等待)
        self.action_space = spaces.Discrete(num_nodes + 2)

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

        # 初始化环境状态
        #self.reset()

    def reset(self, seed = 2024) -> Dict[str, np.ndarray]:
        """
        重置环境到初始状态
        返回初始观察值
        """
        dataset_config = {
        'dataset': 'philly_gen',
        'arrival_rate': 60,  # 每时隙20个任务
        'cv_factor': 1.0,
        'total_jobs': 100,
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
        self._load_jobs()
        return self._get_observation()

    def step(self, action: int) -> tuple[Dict[str, np.ndarray], float, bool, dict]: 
        """
        执行一个动作
        
        参数:
            action: 调度动作 (整数)
            
        返回:
            observation: 新的状态观察值
            reward: 获得的奖励
            done: 是否终止
            info: 附加信息        
        """
        
        # 加载新到达的任务到队列
        self._load_jobs()

        # 终止条件检查
        if self._is_done():
            return self._get_observation(), 0.0, True, {}
        
        # 处理空队列情况
        if not self.visible_jobs:
            #self.current_time += 1
            self._update_active_jobs()
            return self._get_observation(), self.reward, False, {}

        # 获取当前待处理的任务
        current_job = self.visible_jobs[0]
        done = False

        # 解析动作类型
        if action <= self.num_nodes - 1:  # 分配到服务器
            node_id = action
            allocated = min(
                current_job.num_gpus - sum(current_job.assigned_gpus),
                self.cluster_state[node_id]
            )
            
            # 更新集群状态和任务分配
            self.cluster_state[node_id] -= allocated
            current_job.assigned_gpus[node_id] += allocated

            required_gpus = current_job.num_gpus - sum(current_job.assigned_gpus)

            
            # 检查是否分配完成
            if required_gpus <= 0:
                self._start_job_execution(current_job)
            else:  # 分配尚未完成，放到下一轮处理，需要进行惩罚
                self.wait_jobs.append(current_job)  
                self.visible_jobs.pop(0)
                self.reward -=  required_gpus * 5  # 剩余的所需GPU越多，惩罚越大


        elif action == self.num_nodes:    # 分配到云
            if not current_job.privacy:
                current_job.start = self.current_time
                current_job.state = 'CLOUD'
                cloud_cost = current_job.cost
                self.cloud_cost.append(cloud_cost)
                self.total_cloud_cost += cloud_cost
                self.reward -= self.cloud_cost_weight * cloud_cost
                self.finished_jobs.append(current_job)
                self.visible_jobs.pop(0)
            else:  # 如果把隐私任务分到云端则需要加一个很大的惩罚，并且重新分配
                self.wait_jobs.append(current_job)  
                self.visible_jobs.pop(0)
                self.reward -=  500


        elif action == self.num_nodes + 1:  # 等待
            if not self.visible_jobs: # 如果是没有可见任务进行的等待，则不进行任何操作
                pass
            else: 
                required_gpus = current_job.num_gpus - sum(current_job.assigned_gpus)
                self.wait_jobs.append(current_job)
                self.visible_jobs.pop(0)
                # 区分是资源不足导致的等待（给小惩罚）还是有资源时主动等待（给一个大的惩罚）
                if all(x == 0 for x in self.cluster_state) or (sum(self.cluster_state) < required_gpus): # 如果是资源不足导致任务等待，给一个小惩罚
                    self.reward -= 5
                else:
                    self.reward -= 500
                    #self.reward -= ((self.current_time + 1) / (current_job.timeslot + 1)) * 5  #  选择一次等待就会给出惩罚，等待越久惩罚越大（这个5也是随便设置的）

        # 不进行时间推进推进模拟时间并更新任务状态
        #self.current_time += 1
        #self._update_active_jobs()

        # 验证语句
        #print(f"Finished: {len(self.finished_jobs)}/50")

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
        self.visible_jobs.pop(0)
        self.reward += ((job.timeslot + 1) / (job.start + 1)) * 5   # 在启动执行时，给一个正奖励函数，越早开始执行奖励越大（这个5是随便设置的）
        self.reward -= used_nodes_minus_1  # 奖励再减去使用的（节点数-1）

    def _get_actual_time(self, job: Job, inter_nodes_factor=0.1) -> float:
        """计算考虑跨节点惩罚的实际运行时间"""
        used_nodes = sum(gpu > 0 for gpu in job.assigned_gpus)
        if used_nodes == 1:
            return job.runtime, 0
        return job.runtime * (1 + inter_nodes_factor * (used_nodes - 1)), used_nodes - 1

    def _update_active_jobs(self):
        """更新运行中的任务状态"""
        # 把 wait_jobs 的全部元素搬到 queue 前端
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
    
    def get_action_mask(self) -> np.ndarray:
        """
        生成动作掩码，1表示有效动作，0表示无效动作
        注意：假设当前处理队列中的第一个任务（如果有）
        """
        mask = [False] * self.action_space.n
        
        if not self.visible_jobs:
            # 空队列时只能选择等待
            mask[self.num_nodes + 1] = True
            return mask
        
        current_job = self.visible_jobs[0]
        required_gpus = current_job.num_gpus - sum(current_job.assigned_gpus)
        
        # 检查服务器节点选项
        for node_id in range(self.num_nodes):
            if self.cluster_state[node_id] >= 1:
                mask[node_id] = True
                
        # 检查云选项（需任务不要求隐私）
        if not current_job.privacy:
            mask[self.num_nodes] = True
            
        # 等待动作始终有效
        mask[self.num_nodes + 1] = True
        
        return mask
    
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