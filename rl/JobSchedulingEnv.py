import numpy as np
import tensorflow as tf
from typing import Any, Dict, List, Optional
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step
from job import Job
import copy


class JobSchedulingEnv(py_environment.PyEnvironment):
    def __init__(self, total_jobs: List[Job], num_nodes=3, num_gpus_per_node=8, cloud_cost_weight=1):
        super().__init__()
        # 动作空间：选择放置位置 (0~N-1: 服务器, N: 云, N+1: 等待)
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, 
            minimum=0, maximum=num_nodes + 1, 
            name='placement'
        )  # 设置了动作空间的大小
        
        # 状态空间定义
        self._state_spec = {
            'jobs': array_spec.ArraySpec(
                shape=(None, 5),  # 可变长度任务序列
                dtype=np.float32,
                name='job_features'
            ),
            'cluster': array_spec.ArraySpec(
                shape=(num_nodes,),
                dtype=np.int32,
                name='cluster_state'
            )
        }
        
        # 环境参数
        self.num_nodes = num_nodes
        self.num_gpus_per_node = num_gpus_per_node
        self.cloud_cost_weight = cloud_cost_weight
        self.total_jobs = copy.deepcopy(total_jobs)
        
        # 重置环境
        self.reset()
        

    def _reset(self):
        """初始化环境状态"""
        # 集群初始状态：每个服务器的可用GPU数
        self.cluster_state = np.full(
            shape=(self.num_nodes,),
            fill_value=self.num_gpus_per_node,
            dtype=np.int32
        )
        
        # 任务队列（等待调度的任务）
        self.queue = []
        self.active_jobs = []
        self.finished_jobs = []
        self.current_time = 0
        self.total_cloud_cost = 0.0
        self.reward = 0.0
        
        return time_step.restart(self._get_observation())
    
    def _load_jobs(self):
        """将任务按时隙加载到队列中"""
        # 根据任务的 `timeslot` 信息，将任务按到达时隙分配到队列
        for job in self.total_jobs:
            if job.timeslot >= self.current_time:
                self.queue.append(job)
                self.total_jobs.remove(job)

    def get_actual_time(job, inter_nodes_factor = 0.1):  # 根据安排的服务器数量调整实际运行的时间
        num_nodes_used = sum(1 for gpu_count in job.assigned_gpus if gpu_count > 0)  # 任务分配的服务器数量
        if num_nodes_used == 1:
            return job.runtime
        else:
            penalty_factor = 1 + inter_nodes_factor * (num_nodes_used - 1)  # 每多跨一个服务器增加inter_nodes_factor时间，默认是10%
            return job.runtime * penalty_factor
        
    def _step(self, action):
        """执行动作"""
        # 选择要处理的任务（简化版：处理队列中第一个任务）
        self._load_jobs()

        # 如果当前队列为空且没有等待的任务，则直接终止
        if len(self.total_jobs) == 0 and len(self.queue) == 0 and len(self.active_jobs) == 0:
            return time_step.termination(self._get_observation(), 0.0)
        
        # 如果当前队列为空，但还有未到达的任务，直接推进时间步
        if len(self.queue) == 0:
            self.current_time += 1
            self._update_active_jobs()
            return time_step.transition(
            self._get_observation(), 
            reward=self.reward,
            discount=0.99
        )
        
        # 如果当前队列不空，则对任务进行调度
        current_job = self.queue[0]
        # self.current_time = current_job.timeslot
        done = False
        
        # 解析动作
        if action <= self.num_nodes - 1:  # 分配到服务器
            node_id = action  # 记录选择的服务器
            allocated_gpus_num = sum(current_job.assigned_gpus)
            allocated = min(
                current_job.num_gpus - allocated_gpus_num, 
                self.cluster_state[node_id]
            )  # 取当前需求的GPU和服务器现有GPU中较小的那个
            self.cluster_state[node_id] -= allocated
            current_job.assigned_gpu[node_id] += allocated
            
            
            if allocated_gpus_num == current_job.num_gpus:
                # 开始执行
                current_job.start = self.current_time
                current_job.state = 'LOCAL'
                current_job.actual_time = self.get_actual_time(current_job)  # 设置实际运行的时间
                self.active_jobs.append(current_job)
                self.queue.pop(0)
                    
        elif action == self.num_nodes:  # 分配到云
            if not current_job.privacy:
                current_job.state = 'CLOUD'
                current_job.start = self.current_time
                # 计算云费用
                cloud_cost = current_job.cost
                self.total_cloud_cost += cloud_cost
                self.reward -= self.cloud_cost_weight * cloud_cost
                self.finished_jobs.append(current_job)
                self.queue.pop(0)
            #?? # else？要考虑隐私性吗，能否使用mask技术在隐私时把这个动作屏蔽掉
                
        elif action == self.num_nodes + 1:  # 等待
            pass  # 不做处理，留在队列
        
        # 更新时间并计算奖励
        self.current_time += 1  # 时间步要与时隙长度对应起来
        self._update_active_jobs()
        
        # 检查是否终止，应该改成所有任务完成时才终止，在测试时可以自定义数据
        #done = (len(self.queue) == 0 and len(self.total_jobs) == 0 and len(self.active_jobs) == 0)
        # 测试，完成50个任务就停止
        done = len(self.finished_jobs) >= 50 
        
        return time_step.transition(
            self._get_observation(), 
            reward=self.reward,
            discount=0.99
        ) if not done else time_step.termination(
            self._get_observation(), 
            self.reward
        )

    def _get_observation(self):
        """构建状态观察"""
        # 任务特征矩阵 (5维特征)
        job_features = []
        for job in self.queue + self.active_jobs:
            feat = [
                job.idx,
                job.arrival,
                float(job.privacy),
                job.runtime,
                job.num_gpus - sum(job.assigned_gpus)  # 剩余需要的GPU
            ]
            job_features.append(feat)
        
        return {
            'jobs': np.array(job_features, dtype=np.float32),
            'cluster': self.cluster_state.copy()
        }

    def _update_active_jobs(self):
        """更新运行中的任务状态"""
        remaining_jobs = []
        for job in self.active_jobs:
            job.actual_time -= 20 / 60  # 运行了一个时隙(20分钟)，转为小时为单位
            if job.actual_time <= 0:  # 运行完成后
                # 释放GPU资源
                for node_id in range(self.num_nodes):
                    self.cluster_state[node_id] += job.assigned_gpus[node_id]
                self.finished_jobs.append(job)
                self.reward -= (job.actual_time + (job.start - job.arrival) * 20 / 60)  # JCT作为负奖励
            else:
                remaining_jobs.append(job)
        self.active_jobs = remaining_jobs