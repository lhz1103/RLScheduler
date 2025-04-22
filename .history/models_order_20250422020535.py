import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from gym import spaces

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, max_jobs = 20, hidden_dim=64):  # max_jobs就是Env里的max_jobs_per_ts
        """
        Actor网络定义
        :param state_dim: 状态维度 (需处理字典状态)
        :param action_dim: 动作空间维度
        :param hidden_dim: 隐藏层维度
        """
        super().__init__()
        self.max_jobs = max_jobs  # 保存最大任务数

         # 任务特征编码器
        self.job_encoder = nn.Sequential(
            nn.Linear(5, 16),  # 将5维任务特征映射到16维
            nn.LayerNorm(16),  # 使用LayerNorm替代BatchNorm
            nn.ReLU()
        )

        # 集群特征编码器
        self.cluster_encoder = nn.Sequential(
            nn.Linear(state_dim['cluster'][0], 16),  # 集群状态编码
            nn.LayerNorm(16),
            nn.ReLU()
        )
        
        # 联合特征提取层
        self.shared_layers = nn.Sequential(
            nn.Linear(16 * max_jobs + 16, hidden_dim),  # 拼接所有任务特征和集群特征
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 动作输出层修改为排序分数
        self.priority_head = nn.Sequential(
            nn.Linear(hidden_dim, max_jobs),  # 输出每个任务的优先级分数
            nn.Sigmoid()  # 将分数限制在0-1范围
        )
        
    def forward(self, state_dict):
        """
        处理字典形式的状态输入:
        state_dict = {
            'jobs': [batch_size, num_jobs, 5],
            'cluster': [batch_size, num_nodes]
        }
        """
         # 处理任务特征
        jobs = state_dict['jobs']  # [batch, max_jobs, 5]
        batch_size = jobs.size(0)

        # 编码每个任务特征
        job_feats = self.job_encoder(jobs.view(-1, 5))  # [batch * max_jobs, 16]
        job_feats = job_feats.view(batch_size, self.max_jobs, -1)  # [batch, max_jobs, 16]

        # 拼接所有任务特征
        job_combined = job_feats.view(batch_size, -1)  # [batch, max_jobs * 16]

        # 处理集群特征
        cluster = self.cluster_encoder(state_dict['cluster'].float())  # [batch, 16]

        # 联合特征
        combined = torch.cat([job_combined, cluster], dim=1)  # [batch, max_jobs * 16 + 16]

        # 输出动作logits
        return self.priority_head(self.shared_layers(combined))

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, max_jobs = 20, hidden_dim=64):
        """
        Critic网络定义
        :param state_dim: 状态维度
        :param hidden_dim: 隐藏层维度
        """
        super().__init__()
        self.max_jobs = max_jobs
        # 状态特征处理层（与Actor共享结构）
        self.job_encoder = nn.Sequential(
            nn.Linear(5, 16),
            #nn.BatchNorm1d(16),
            nn.LayerNorm(16),  # 替换为 LayerNorm
            nn.ReLU()
        )
        
        # 集群特征编码器
        self.cluster_encoder = nn.Sequential(
            nn.Linear(state_dim['cluster'][0], 16),  # 集群状态编码
            nn.LayerNorm(16),
            nn.ReLU()
        )

        # 价值估计层
        self.value_layers = nn.Sequential(
            nn.Linear(16 * max_jobs + 16, hidden_dim),  # 拼接所有任务特征和集群特征
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # 输出单个价值估计
        )
        
    def forward(self, state_dict):
        """
        处理字典形式的状态输入:
        state_dict = {
            'jobs': [batch_size, max_jobs, 5],  # 固定维度
            'cluster': [batch_size, num_nodes]  # 集群状态
        }
        """
        # 处理任务特征
        jobs = state_dict['jobs']  # [batch, max_jobs, 5]
        batch_size = jobs.size(0)

        # 编码每个任务特征
        job_feats = self.job_encoder(jobs.view(-1, 5))  # [batch * max_jobs, 16]
        job_feats = job_feats.view(batch_size, self.max_jobs, -1)  # [batch, max_jobs, 16]

        # 拼接所有任务特征
        job_combined = job_feats.view(batch_size, -1)  # [batch, max_jobs * 16]

        # 处理集群特征
        cluster = self.cluster_encoder(state_dict['cluster'].float())  # [batch, 16]

        # 联合特征
        combined = torch.cat([job_combined, cluster], dim=1)  # [batch, max_jobs * 16 + 16]

        # 输出价值估计
        return self.value_layers(combined)

def create_actor_critic_networks(env, device='cuda'):
    """
    创建Actor-Critic网络对
    :param env: 环境实例（用于获取状态/动作空间信息）
    :param device: 计算设备
    :return: (actor, critic) 网络元组
    """
    # 解析状态空间维度
    state_dim = {
        'jobs': (env.max_jobs_per_ts, 5),  # 固定为最大任务数
        'cluster': (env.num_nodes,)
    }

    # 动作空间维度
    action_dim = env.max_jobs_per_ts

    # 实例化网络并直接部署到设备
    actor = ActorNetwork(
        state_dim=state_dim,
        action_dim=action_dim,
        max_jobs=env.max_jobs_per_ts  # 传递最大任务数
    ).to(device)

    critic = CriticNetwork(
        state_dim=state_dim,
        max_jobs=env.max_jobs_per_ts  # 传递最大任务数
    ).to(device)

    return actor, critic

# 使用示例
if __name__ == "__main__":
    # 假设已创建环境实例
    class DummyEnv:
        action_space = spaces.Discrete(5)
        num_nodes = 3
        max_jobs_per_ts = 20  # 新增属性

    env = DummyEnv()

    # 创建网络
    actor, critic = create_actor_critic_networks(env)

    # 测试前向传播
    dummy_state = {
        'jobs': torch.randn(2, env.max_jobs_per_ts, 5).to(device='cuda'),  # batch_size=2
        'cluster': torch.randint(0, 8, (2, env.num_nodes)).to(device='cuda')
    }

    # Actor输出
    action_logits = actor(dummy_state)
    print("Actor output shape:", action_logits.shape)  # 应输出 [2, 5]

    # Critic输出
    value = critic(dummy_state)
    print("Critic output shape:", value.shape)  # 应输出 [2, 1]