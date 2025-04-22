import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from collections import deque
import random
from job import Job
from job_generator import load_processed_jobs
import matplotlib.pyplot as plt
import os
import pandas as pd

class PPO:
    def __init__(self, env, device='cuda', 
                 actor_lr=3e-4, 
                 critic_lr=1e-3,
                 gamma=0.99,  # 折扣因子
                 gae_lambda=0.95,  # 用于计算广义优势估计（GAE）的平滑因子
                 clip_epsilon=0.2,  # 策略更新时截断比率的阈值，限制新旧策略之间的更新幅度
                 ppo_epochs=4,  # 每次更新时遍历整个经验数据的轮数
                 batch_size=64):  # mini-batch 的大小
        
        self.env = env
        self.device = device
        
        # 超参数设置
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        
        # 初始化网络
        self.actor, self.critic = create_actor_critic_networks(env, device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # 经验缓存
        self.buffer = deque(maxlen=2048)

        
        
    def preprocess_state(self, state):
        """正确处理可变长度任务填充"""
        # 任务特征填充
        job_feats = np.zeros((self.env.max_jobs_per_ts, 5), dtype=np.float32)
        visible_jobs = min(len(state['jobs']), self.env.max_jobs_per_ts)
        job_feats[:visible_jobs] = state['jobs'][:visible_jobs]
        
        return {
            'jobs': torch.FloatTensor(job_feats).unsqueeze(0).to(self.device),  # 添加批次维度
            'cluster': torch.IntTensor(state['cluster']).unsqueeze(0).to(self.device)
        }
    
    def get_action(self, state, action_mask):
        """
        带掩码的动作选择
        返回动作、对数概率、价值估计
        """
        with torch.no_grad():
            state_tensor = self.preprocess_state(state)
            logits = self.actor(state_tensor)
            
            # 应用动作掩码
            mask_tensor = torch.tensor(np.array(action_mask), dtype=torch.bool, device=self.device).unsqueeze(0)  # 构建掩码张量
            #logits[~mask_tensor] = -float('inf') # 将不允许选择的动作对应的 logits 设为负无穷，使得后续 softmax 后概率趋近于 0
            
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            action = dist.sample()
            
            log_prob = dist.log_prob(action)
            value = self.critic(state_tensor)
            
        return action.item(), log_prob.item(), value.item()
    
    def compute_gae(self, rewards, values, dones):
        """计算广义优势估计
            利用时间差分误差（TD误差）和衰减因子递归计算每一步的优势值，既考虑即时奖励，也纳入未来奖励的影响
        """
        advantages = []
        last_advantage = 0
        next_value = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                last_advantage = delta
            else:
                delta = rewards[t] + self.gamma * next_value - values[t]
                last_advantage = delta + self.gamma * self.gae_lambda * last_advantage
            
            advantages.append(last_advantage)
            next_value = values[t]
        
        advantages = advantages[::-1]
        return torch.tensor(advantages, dtype=torch.float32).to(self.device)
    
    def update(self):
        """执行PPO更新"""
        # 将缓存数据转换为张量
        states, actions, old_log_probs, rewards, dones, masks = zip(*self.buffer)
        
        # 转换为PyTorch张量
        states = [self.preprocess_state(s) for s in states]
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        masks_np = np.array(masks, dtype=bool)  # 先转换为numpy数组
        masks = torch.from_numpy(masks_np).to(self.device)  # 再转换为tensor
        
        # 计算优势
        with torch.no_grad():
            values = torch.stack([self.critic(s) for s in states]).squeeze()
        advantages = self.compute_gae(rewards, values, dones)
        returns = advantages + values
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 数据索引
        indices = np.arange(len(states))
        
        # 训练多个epoch
        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)
            
            # 分批次训练
            for start in range(0, len(indices), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_states = [states[i] for i in batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_masks = masks[batch_indices].unsqueeze(1)  # 添加一个维度以匹配logits形状
                
                # 计算新策略的概率
                current_logits = torch.stack([self.actor(s) for s in batch_states])
                current_logits[~batch_masks] = -float('inf')  # 应用掩码
                current_probs = F.softmax(current_logits, dim=-1)
                dist = Categorical(current_probs)
                current_log_probs = dist.log_prob(batch_actions)
                
                # 计算比率
                ratios = torch.exp(current_log_probs - batch_old_log_probs)
                
                # 计算策略损失
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1-self.clip_epsilon, 1+self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 计算价值损失
                #current_values = torch.stack([self.critic(s) for s in batch_states]).squeeze()
                #value_loss = F.mse_loss(current_values, batch_returns)
                current_values = torch.stack([self.critic(s).view(-1) for s in batch_states])  # 确保形状为 [batch_size, 1]
                batch_returns = batch_returns.view(-1, 1)  # 匹配维度
                value_loss = F.mse_loss(current_values, batch_returns)
                
                # 总损失
                loss = policy_loss + 0.5 * value_loss
                
                # 反向传播
                self.actor_optim.zero_grad()
                self.critic_optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.actor_optim.step()
                self.critic_optim.step()
        
        # 清空缓存
        self.buffer.clear()

    def plot_train_metrics(self, episode_rewards, queue_lengths, visible_jobs_counts, active_jobs_counts):
        fig_dir = os.path.expanduser("~/starburst/RLscheduler/figures")
        os.makedirs(fig_dir, exist_ok=True)
        plt.figure(figsize=(15, 10))
    
        # 奖励曲线
        plt.subplot(2, 2, 1)
        plt.plot(episode_rewards, color='tab:blue', alpha=0.5)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
    
        # 队列长度
        plt.subplot(2, 2, 2)
        plt.plot(queue_lengths, color='tab:orange', alpha=0.5)
        plt.title('Queue Length Over Time')
        plt.xlabel('Timestep')
        plt.ylabel('Queue Length')
    
        # 可见任务数
        plt.subplot(2, 2, 3)
        plt.plot(visible_jobs_counts, color='tab:green', alpha=0.5)
        plt.title('Visible Jobs Over Time')
        plt.xlabel('Timestep')
        plt.ylabel('Visible Jobs')
    
        # 活动任务数
        plt.subplot(2, 2, 4)
        plt.plot(active_jobs_counts, color='tab:red', alpha=0.5)
        plt.title('Active Jobs Over Time')
        plt.xlabel('Timestep')
        plt.ylabel('Active Jobs')
    
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'training_metrics.png'))
        plt.close()

        # 保存数据到excel表格
        episode_numbers = list(range(1, len(episode_rewards) + 1))

        # Mean JCT
        df_rewards = pd.DataFrame([episode_rewards], columns=episode_numbers, index=['RL'])
        df_rewards.to_excel(os.path.join(fig_dir, 'Rewards.xlsx'))
    
    def plot_jct_statistics(self, episode_jct_mean, episode_jct_p95, episode_jct_p99):
        fig_dir = os.path.expanduser("~/starburst/RLscheduler/figures")
        os.makedirs(fig_dir, exist_ok=True)
        plt.figure(figsize=(15, 10))

        # 第一个子图：Episode Mean JCT
        plt.subplot(3, 1, 1)
        plt.plot(episode_jct_mean, color='tab:blue', alpha=0.7)
        plt.title('Episode Mean JCT')
        plt.xlabel('Episode')
        plt.ylabel('Mean JCT/hours')

        # 第二个子图：Episode P95 JCT
        plt.subplot(3, 1, 2)
        plt.plot(episode_jct_p95, color='tab:orange', alpha=0.7)
        plt.title('Episode P95 JCT')
        plt.xlabel('Episode')
        plt.ylabel('P95 JCT/hours')

        # 第三个子图：Episode P99 JCT
        plt.subplot(3, 1, 3)
        plt.plot(episode_jct_p99, color='tab:red', alpha=0.7)
        plt.title('Episode P99 JCT')
        plt.xlabel('Episode')
        plt.ylabel('P99 JCT/hours')

        # 自动调整子图间距
        plt.tight_layout()

        # 保存图片
        plt.savefig(os.path.join(fig_dir, 'JCT.png'))
        plt.close()

        # 保存数据到excel表格
        episode_numbers = list(range(1, len(episode_jct_mean) + 1))

        # Mean JCT
        df_mean = pd.DataFrame([episode_jct_mean], columns=episode_numbers, index=['RL'])
        df_mean.to_excel(os.path.join(fig_dir, 'Mean_JCT_hours.xlsx'))

        # P95 JCT
        df_p95 = pd.DataFrame([episode_jct_p95], columns=episode_numbers, index=['RL'])
        df_p95.to_excel(os.path.join(fig_dir, 'P95_JCT_hours.xlsx'))

        # P99 JCT
        df_p99 = pd.DataFrame([episode_jct_p99], columns=episode_numbers, index=['RL'])
        df_p99.to_excel(os.path.join(fig_dir, 'P99_JCT_hours.xlsx'))
    
    def plot_wait_time_statistics(self, episode_wait_time_mean, episode_wait_time_p95, episode_wait_time_p99):
        fig_dir = os.path.expanduser("~/starburst/RLscheduler/figures")
        os.makedirs(fig_dir, exist_ok=True)
        plt.figure(figsize=(15, 10))

        # 第一个子图：Episode Mean Wait Time
        plt.subplot(3, 1, 1)
        plt.plot(episode_wait_time_mean, color='tab:blue', alpha=0.7)
        plt.title('Episode Mean Wait Time')
        plt.xlabel('Episode')
        plt.ylabel('Mean Wait Time/hours')

        # 第二个子图：Episode P95 Wait Time
        plt.subplot(3, 1, 2)
        plt.plot(episode_wait_time_p95, color='tab:orange', alpha=0.7)
        plt.title('Episode P95 Wait Time')
        plt.xlabel('Episode')
        plt.ylabel('P95 Wait Time/hours')

        # 第三个子图：Episode P99 Wait Time
        plt.subplot(3, 1, 3)
        plt.plot(episode_wait_time_p99, color='tab:red', alpha=0.7)
        plt.title('Episode P99 Wait Time')
        plt.xlabel('Episode')
        plt.ylabel('P99 Wait Time/hours')

        # 自动调整子图间距
        plt.tight_layout()

        # 保存图片
        plt.savefig(os.path.join(fig_dir, 'Wait_Time.png'))
        plt.close()

        # 保存数据到excel表格
        episode_numbers = list(range(1, len(episode_wait_time_mean) + 1))

        # Mean Wait Time
        df_mean = pd.DataFrame([episode_wait_time_mean], columns=episode_numbers, index=['RL'])
        df_mean.to_excel(os.path.join(fig_dir, 'Mean_Wait_Time_hours.xlsx'))

        # P95 Wait Time
        df_p95 = pd.DataFrame([episode_wait_time_p95], columns=episode_numbers, index=['RL'])
        df_p95.to_excel(os.path.join(fig_dir, 'P95_Wait_Time_hours.xlsx'))

        # P99 Wait Time
        df_p99 = pd.DataFrame([episode_wait_time_p99], columns=episode_numbers, index=['RL'])
        df_p99.to_excel(os.path.join(fig_dir, 'P99_Wait_Time_hours.xlsx'))

    def plot_cloud_cost_statistics(self, episode_cloud_cost_mean, episode_cloud_cost_p95, episode_cloud_cost_p99):
        fig_dir = os.path.expanduser("~/starburst/RLscheduler/figures")
        os.makedirs(fig_dir, exist_ok=True)
        plt.figure(figsize=(15, 10))

        # 第一个子图：Episode Mean Wait Time
        plt.subplot(3, 1, 1)
        plt.plot(episode_cloud_cost_mean, color='tab:blue', alpha=0.7)
        plt.title('Episode Mean Cloud Cost')
        plt.xlabel('Episode')
        plt.ylabel('Mean Cloud Cost')

        # 第二个子图：Episode P95 Wait Time
        plt.subplot(3, 1, 2)
        plt.plot(episode_cloud_cost_p95, color='tab:orange', alpha=0.7)
        plt.title('Episode P95 Cloud Cost')
        plt.xlabel('Episode')
        plt.ylabel('P95 Cloud Cost')

        # 第三个子图：Episode P99 Wait Time
        plt.subplot(3, 1, 3)
        plt.plot(episode_cloud_cost_p99, color='tab:red', alpha=0.7)
        plt.title('Episode P99 Cloud Cost')
        plt.xlabel('Episode')
        plt.ylabel('P99 Cloud Cost')

        # 自动调整子图间距
        plt.tight_layout()

        # 保存图片
        plt.savefig(os.path.join(fig_dir, 'Cloud_Cost.png'))
        plt.close()

        # 保存数据到excel表格
        episode_numbers = list(range(1, len(episode_cloud_cost_mean) + 1))

        # Mean Cloud Cost
        df_mean = pd.DataFrame([episode_cloud_cost_mean], columns=episode_numbers, index=['RL'])
        df_mean.to_excel(os.path.join(fig_dir, 'Mean_Cloud_Cost.xlsx'))

        # P95 Cloud Cost
        df_p95 = pd.DataFrame([episode_cloud_cost_p95], columns=episode_numbers, index=['RL'])
        df_p95.to_excel(os.path.join(fig_dir, 'P95_Cloud_Cost.xlsx'))

        # P99 Cloud Cost
        df_p99 = pd.DataFrame([episode_cloud_cost_p99], columns=episode_numbers, index=['RL'])
        df_p99.to_excel(os.path.join(fig_dir, 'P99_Cloud_Cost.xlsx'))
    
    def append_method_to_excel(file_path, method_name, data_row):
        '''往excel里添加其他方法的数据'''
        # Episode 列表自动从数据长度生成
        episode_numbers = list(range(1, len(data_row) + 1))
    
        # 创建新的 DataFrame 行
        new_row_df = pd.DataFrame([data_row], columns=episode_numbers, index=[method_name])
    
        if os.path.exists(file_path):
            # 文件存在，读取原始文件
            existing_df = pd.read_excel(file_path, index_col=0)
        
            # 合并原始数据和新数据
            updated_df = pd.concat([existing_df, new_row_df])
        else:
            # 文件不存在，直接使用新数据
            updated_df = new_row_df

        # 写回 Excel（会覆盖原文件）
        updated_df.to_excel(file_path)

    
    def train(self, max_episodes=1000, update_interval=2048):
        """训练主循环"""
        episode_rewards = []
        queue_lengths = []
        visible_jobs_counts = []
        active_jobs_counts = []
        #timesteps = 0

        # 统计数据
        episode_jct_mean = []
        episode_jct_p95 = []
        episode_jct_p99 = []

        episode_wait_time_mean = []
        episode_wait_time_p95 = []
        episode_wait_time_p99 = []

        episode_cloud_cost_mean = []
        episode_cloud_cost_p95 = []
        episode_cloud_cost_p99 = []

        
        for episode in range(max_episodes):

            # 验证语句
            print(f"\n=== Starting Episode {episode+1} ===")
            print(f"Initial Visible Jobs: {len(self.env.visible_jobs)}")
            print(f"Total Jobs Remaining: {len(self.env.total_jobs)}")
            
            # 每个Episode都重置环境和初始化状态，也就是说每个Episode都是一整个调度过程（直到done）
            state = self.env.reset(seed = 2024 + episode)
            done = False
            episode_reward = 0
            episode_steps = 0
            
            while not done:
                self.env._load_jobs()
                # 收集经验
                #for _ in range(update_interval):
                while self.env.visible_jobs:  # 处理当前时隙所有可见任务
                    action_mask = self.env.get_action_mask()
                    action, log_prob, value = self.get_action(state, action_mask)
                    
                    next_state, reward, done, _ = self.env.step(action)
                    
                    # 收集指标
                    queue_lengths.append(len(self.env.queue))
                    visible_jobs_counts.append(len(self.env.visible_jobs))
                    active_jobs_counts.append(len(self.env.active_jobs))
                    
                    episode_steps += 1
                    #timesteps += 1
                    
                    # 存储转换
                    self.buffer.append((
                        state,
                        action,
                        log_prob,
                        reward,
                        done,
                        action_mask
                    ))

                    # 验证语句
                    print(f"Buffer size: {len(self.buffer)}/{self.batch_size}")
                    
                    
                    state = next_state
                    if done:
                        break

                

                # 更新网络
                if len(self.buffer) >= self.batch_size:
                    self.update()
                
                print(f"Timeslot: {self.env.current_time}")
                
                # 推进时间并更新任务状态
                self.env.current_time += 1
                self.env._update_active_jobs()
                
                done = self.env._is_done()  # 在外层循环外还得再调用一次，否则在visible_jobs为空，但还有任务没结束时，done无法正确结束
                

                # 验证语句
                print(f"Finished: {len(self.env.finished_jobs)}/100")
            
            # 记录统计数据
            jct_mean, jct_p95, jct_p99, wait_time_mean, wait_time_p95, wait_time_p99 = self.env.get_time_statistics()
            episode_jct_mean.append(jct_mean)
            episode_jct_p95.append(jct_p95)
            episode_jct_p99.append(jct_p99)

            episode_wait_time_mean.append(wait_time_mean)
            episode_wait_time_p95.append(wait_time_p95)
            episode_wait_time_p99.append(wait_time_p99)
            
            # 在费用方面，这里只考虑了上传到云端的任务，也就是说数量较少，有可能出现mean=p95=p99的情况（只有一个任务在云端）
            cloud_cost_mean, cloud_cost_p95, cloud_cost_p99 = self.env.get_cost_statistics()
            episode_cloud_cost_mean.append(cloud_cost_mean)
            episode_cloud_cost_p95.append(cloud_cost_p95)
            episode_cloud_cost_p99.append(cloud_cost_p99)
            # 记录训练进度
            episode_reward += reward
            episode_rewards.append(episode_reward)
            print(
            f"Ep {episode+1:3d} | "
            f"Reward: {episode_reward:7.2f} | "
            f"Queue: {np.mean(queue_lengths[-episode_steps:]):5.2f} | "
            f"Visible: {np.mean(visible_jobs_counts[-episode_steps:]):3.2f} | "
            f"Active: {np.mean(active_jobs_counts[-episode_steps:]):3.2f}"
            )
            
            # 定期保存模型
            today = datetime.now().strftime("%Y-%m-%d")
            save_dir = os.path.expanduser(f"~/starburst/RLscheduler/checkpoints/{today}")  # ← 你的目标目录
            # 定期保存模型
            if (episode+1) % 50 == 0:
                torch.save(self.actor.state_dict(), f"ppo_actor_{episode+1}.pth")
                torch.save(self.critic.state_dict(), f"ppo_critic_{episode+1}.pth")
        
        # 训练结束后绘图
        self.plot_train_metrics(
            episode_rewards,
            queue_lengths,
            visible_jobs_counts,
            active_jobs_counts
        )
        self.plot_jct_statistics(
            episode_jct_mean,
            episode_jct_p95,
            episode_jct_p99
        )
        self.plot_wait_time_statistics(
            episode_wait_time_mean,
            episode_wait_time_p95,
            episode_wait_time_p99
        )
        self.plot_cloud_cost_statistics(
            episode_cloud_cost_mean,
            episode_cloud_cost_p95,
            episode_cloud_cost_p99
        )
        return episode_rewards
    
    

# 使用示例
if __name__ == "__main__":
    # 假设已实现Job类和JobSchedulingEnv
    from JobSchedulingEnv import JobSchedulingEnv
    from models import create_actor_critic_networks
    
    # 创建环境
    num_nodes = 16
    env = JobSchedulingEnv(num_nodes=num_nodes)
    
    # 初始化PPO
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ppo = PPO(env, device=device)
    
    # 开始训练
    rewards = ppo.train(max_episodes=500)