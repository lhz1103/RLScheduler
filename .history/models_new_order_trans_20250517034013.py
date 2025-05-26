import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from gym import spaces
"""
5.16：
1. 在状态中删除id、到达时间和隐私，并添加cost；
2. 把状态token从映射到16到改到32维；
3. 把输入状态拼成一个大矩阵（矩阵维度是正确的吗？），输出到6层8头的transformer编码器

5.17：
1. 试试4层transformer编码器
"""

class ActorNetwork(nn.Module):
    def __init__(self, num_nodes: int, num_gpus: int, max_jobs: int, hidden_dim: int=256, num_heads: int = 8, embed_dim: int = 32, num_layers: int = 4):  # max_jobs就是Env里的max_jobs_per_ts
        """
        Actor网络定义
        :param state_dim: 状态维度 (需处理字典状态)
        :param action_dim: 动作空间维度
        :param hidden_dim: 隐藏层维度
        """
        super().__init__()
        self.max_jobs = max_jobs  # 保存最大任务数
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

         # 任务特征编码器
        self.job_encoder = nn.Sequential(
            nn.Linear(3, embed_dim),  # 将3维任务特征映射到16维
            nn.LayerNorm(embed_dim),  # 使用LayerNorm替代BatchNorm
            nn.ReLU(),
        )

        # 集群特征编码器
        self.cluster_encoder = nn.Sequential(
            nn.Linear(num_nodes, embed_dim),  # 集群状态编码
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )

        # --- gpu_left 编码 (num_nodes*num_gpus)→16 ---
        self.gpu_left_encoder = nn.Sequential(
            nn.Linear(num_nodes * num_gpus, embed_dim), 
            nn.ReLU(),)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True,           # 让输入兼容 (B, seq, E)
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.output_head = nn.Linear(embed_dim, 1)
        
        # 任务 token ↔ cluster/gpu token 的 cross‑attention
        self.cross_attns = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=16, num_heads=num_heads, batch_first=True)
            for _ in range(4)
        ])
        self.cross_norms = nn.ModuleList([
            nn.LayerNorm(16)
            for _ in range(4)
        ])
        
        # 联合特征提取层
        # 拼接后 token 数量翻倍 → 输入维度也翻倍
        in_dim = 16 * max_jobs * 2     # (2*max_jobs, 16) → 展平
        self.shared_layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
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
        jobs = state_dict['jobs']  # [batch, max_jobs, 3]
        cluster = state_dict["cluster"].float()  #  (B, num_nodes)
        gpu_left = state_dict["gpu_left"].float()  # # (B, num_nodes, num_gpus)
        
        B = jobs.size(0)

        # 三部分编码
        job_tokens = self.job_encoder(jobs.view(-1, 3))          
        job_tokens = job_tokens.view(B, self.max_jobs, self.embed_dim)       

        clu_token = self.cluster_encoder(cluster).unsqueeze(1)    
        gpu_token = self.gpu_left_encoder(gpu_left.flatten(1)).unsqueeze(1)  

        # 拼成一个大序列：jobs + [clu, gpu]
        x = torch.cat([job_tokens, clu_token, gpu_token], dim=1)  # (B, J+2, E)           

        # --- Transformer 编码 ---
        x = self.transformer(x)  # (B, J+2, E)

        # --- 输出 job 维度上的分数 ---
        job_out = x[:, :self.max_jobs, :]          # (B, J, E)
        logits  = self.output_head(job_out)        # (B, J, 1)
        logits  = logits.squeeze(-1)               # (B, J)
        return logits

        

class CriticNetwork(nn.Module):
    def __init__(self, num_nodes: int, num_gpus: int, max_jobs: int, hidden_dim: int=256, num_heads: int = 8, embed_dim: int = 32, num_layers: int = 4):
        """
        Critic网络定义
        :param state_dim: 状态维度
        :param hidden_dim: 隐藏层维度
        """
        super().__init__()
        self.max_jobs = max_jobs
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers


        # 1) 三路编码器，将输入都映射到 embed_dim 维
        # 状态特征处理层（与Actor共享结构）
        self.job_encoder = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.LayerNorm(embed_dim),  
            nn.ReLU(),
        )
        
        # 集群特征编码器
        self.cluster_encoder = nn.Sequential(
            nn.Linear(num_nodes, embed_dim),  # 集群状态编码
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )

        self.gpu_left_encoder = nn.Sequential(
            nn.Linear(num_nodes * num_gpus, embed_dim),
            nn.ReLU(),
        )

         # 2) TransformerEncoder，用于融合所有 token
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # 3) 输出头：把所有 token 的输出聚合成一个标量
        #    这里取平均池化 + MLP，也可以用 [CLS] token 或拼接后 MLP
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # 价值估计层
        in_dim = 16 * max_jobs + 16 + 16
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

        # —— 1) 三路编码 —— 
        job_tok = (
            self.job_encoder(jobs.view(-1, 3))
            .view(B, self.max_jobs, self.embed_dim)
        )  # (B, J, E)

        clu_tok = (
            self.cluster_encoder(cluster)
            .unsqueeze(1)
        )  # (B, 1, E)

        gpu_tok = (
            self.gpu_left_encoder(gpu_left.flatten(1))
            .unsqueeze(1)
        )  # (B, 1, E)

        # 拼成一个序列：jobs + [cluster, gpu]
        x = torch.cat([job_tok, clu_tok, gpu_tok], dim=1)  # (B, J+2, E)

        # —— 2) Transformer 编码 —— 
        x = self.transformer(x)  # (B, J+2, E)

        # —— 3) 聚合并输出 —— 
        # 取所有 token 的平均向量
        pooled = x.mean(dim=1)  # (B, E)
        value  = self.value_head(pooled)  # (B, 1)
        return value.squeeze(-1)          # (B,)

def create_actor_critic_networks(env, device='cuda'):
    """
    根据环境参数创建Actor-Critic网络对
    """
    actor = ActorNetwork(
        max_jobs=env.max_jobs_per_ts,
        num_nodes=env.num_nodes,
        num_gpus=env.num_gpus_per_node,
        embed_dim=32,
        num_heads=8,
        num_layers=6
    ).to(device)

    critic = CriticNetwork(
        max_jobs=env.max_jobs_per_ts,
        num_nodes=env.num_nodes,
        num_gpus=env.num_gpus_per_node,
        embed_dim=32,
        num_heads=8,
        num_layers=6
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
        "jobs": torch.randn(2, env.max_jobs_per_ts, 3, device=dev),
        "cluster": torch.randint(0, 9, (2, env.num_nodes), device=dev),
        "gpu_left": torch.zeros(2, env.num_nodes, env.num_gpus_per_node, device=dev),
    }
    print("Actor logits:", actor(dummy_state).shape)   # (2, max_jobs)
    print("Critic value:", critic(dummy_state).shape)  # (2, 1)