import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from gym import spaces

class ActorNetwork(nn.Module):
    def __init__(self, num_nodes: int, num_gpus: int, max_jobs: int, hidden_dim: int=64):  # max_jobs就是Env里的max_jobs_per_ts
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
            nn.ReLU(),
        )

        # 集群特征编码器
        self.cluster_encoder = nn.Sequential(
            nn.Linear(num_nodes, 16),  # 集群状态编码
            nn.LayerNorm(16),
            nn.ReLU(),
        )

        # --- gpu_left 编码 (num_nodes*num_gpus)→32 ---
        self.gpu_left_encoder = nn.Sequential(
            nn.Linear(num_nodes * num_gpus, 32), 
            nn.ReLU(),)
        
        # 联合特征提取层
        in_dim = 16 * max_jobs + 16 + 32
        self.shared_layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),  # 拼接所有任务特征和集群特征
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # 动作输出层修改为排序分数
        self.priority_head = nn.Linear(hidden_dim, max_jobs)  # 输出每个任务的优先级分数
        
        
    def forward(self, state_dict):
        """
        state_dict keys: jobs, cluster, gpu_left
        """
         # 处理任务特征
        jobs = state_dict['jobs']  # [batch, max_jobs, 5]
        cluster = state_dict["cluster"].float()  #  (B, num_nodes)
        gpu_left = state_dict["gpu_left"].float()  # # (B, num_nodes, num_gpus)
        
        B = jobs.size(0)

        job_feat = (
            self.job_encoder(jobs.view(-1, 5))   # (B*max_jobs, 16)
            .view(B, self.max_jobs, -1)
            .flatten(1)                          # (B, 16*max_jobs)
        )

        clu_feat = self.cluster_encoder(cluster)             # (B, 16)
        gpu_feat = self.gpu_left_encoder(gpu_left.flatten(1))# (B, 32)

        x = torch.cat([job_feat, clu_feat, gpu_feat], dim=1)
        return self.priority_head(self.shared_layers(x))            # (B, max_jobs)

        

class CriticNetwork(nn.Module):
    def __init__(self, num_nodes: int, num_gpus: int, max_jobs: int, hidden_dim: int=64):
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
            nn.LayerNorm(16),  # 替换为 LayerNorm
            nn.ReLU(),
        )
        
        # 集群特征编码器
        self.cluster_encoder = nn.Sequential(
            nn.Linear(num_nodes, 16),  # 集群状态编码
            nn.LayerNorm(16),
            nn.ReLU(),
        )

        self.gpu_left_encoder = nn.Sequential(
            nn.Linear(num_nodes * num_gpus, 32),
            nn.ReLU(),
        )

        # 价值估计层
        in_dim = 16 * max_jobs + 16 + 32
        self.value_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),  # 拼接所有任务特征和集群特征
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # 输出单个价值估计
        )
        
    def forward(self, state_dict):
        jobs = state_dict["jobs"]
        cluster = state_dict["cluster"].float()
        gpu_left = state_dict["gpu_left"].float()

        B = jobs.size(0)
        job_feat = (
            self.job_encoder(jobs.view(-1, 5))
            .view(B, self.max_jobs, -1)
            .flatten(1)
        )
        clu_feat = self.cluster_encoder(cluster)
        gpu_feat = self.gpu_left_encoder(gpu_left.flatten(1))

        x = torch.cat([job_feat, clu_feat, gpu_feat], dim=1)
        return self.value_net(x)                 # (B, 1)

def create_actor_critic_networks(env, device='cuda'):
    """
    根据环境参数创建Actor-Critic网络对
    """
    actor = ActorNetwork(
        max_jobs=env.max_jobs_per_ts,
        num_nodes=env.num_nodes,
        num_gpus=env.num_gpus_per_node,
    ).to(device)

    critic = CriticNetwork(
        max_jobs=env.max_jobs_per_ts,
        num_nodes=env.num_nodes,
        num_gpus=env.num_gpus_per_node,
    ).to(device)


    return actor, critic

# 使用示例
if __name__ == "__main__":
    class DummyEnv:
        max_jobs_per_ts = 20
        num_nodes = 4
        num_gpus_per_node = 8

    env = DummyEnv()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor, critic = create_actor_critic_networks(env, dev)

    dummy_state = {
        "jobs": torch.randn(2, env.max_jobs_per_ts, 5, device=dev),
        "cluster": torch.randint(0, 9, (2, env.num_nodes), device=dev),
        "gpu_left": torch.zeros(2, env.num_nodes, env.num_gpus_per_node, device=dev),
    }
    print("Actor logits:", actor(dummy_state).shape)   # (2, max_jobs)
    print("Critic value:", critic(dummy_state).shape)  # (2, 1)