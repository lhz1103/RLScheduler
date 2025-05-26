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
from datetime import datetime
from JobSchedulingEnv_new_order_trans import JobSchedulingEnv
from models_new_order_trans import create_actor_critic_networks
from torch.optim.lr_scheduler import StepLR


class PPO:
    def __init__(self, env, device='cuda', 
                 actor_lr=1e-4, 
                 critic_lr=3e-4,
                 entropy_coef=0.02,          # ★ 初始熵权重
                 lr_decay_step=400,          # ★ 每隔多少 episode 衰减一次 LR
                 lr_decay_gamma=0.7,         # ★ 衰减倍率
                 gamma=0.99,  # 折扣因子
                 gae_lambda=0.95,  # 用于计算广义优势估计（GAE）的平滑因子
                 clip_epsilon=0.2,  # 策略更新时截断比率的阈值，限制新旧策略之间的更新幅度
                 ppo_epochs=3,  # 每次更新时遍历整个经验数据的轮数
                 batch_size=512):  # mini-batch 的大小
        
        self.env = env
        self.device = device
        
        # 超参数设置
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef              # 当前熵权重
        self.entropy_gamma = 0.5                      # 熵权重每次也乘以 γ
        self.lr_decay_step = lr_decay_step
        
        # 初始化网络
        self.actor, self.critic = create_actor_critic_networks(env, device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # 给优化器加调度器，学习率和熵权重随着训练轮数衰减
        self.actor_scheduler  = StepLR(self.actor_optim,
                               step_size=lr_decay_step,
                               gamma=lr_decay_gamma)
        self.critic_scheduler = StepLR(self.critic_optim,
                               step_size=lr_decay_step,
                               gamma=lr_decay_gamma)

        
        # 经验缓存，大小如何控制？
        self.buffer = deque(maxlen=8192)

        # 存储数据路径
        self.figures_path = os.path.expanduser(f"~/starburst/RLscheduler/")

        
        
    def preprocess_state(self, state, to_gpu:bool = True):
        """任务填充"""
        job_feats = np.zeros((self.env.max_jobs_per_ts, 3), dtype=np.float32)
        visible_jobs = min(len(state['jobs']), self.env.max_jobs_per_ts)
        job_feats[:visible_jobs] = state['jobs'][:visible_jobs]

        device = self.device if to_gpu else "cpu"  # 模仿学习时用cpu
        
        return {
            'jobs': torch.tensor(job_feats, dtype=torch.float32, device=device).unsqueeze(0),  # 添加批次维度
            'cluster': torch.tensor(state["cluster"], dtype=torch.float32, device=device).unsqueeze(0),
            'gpu_left': torch.tensor(state["gpu_left"], dtype=torch.float32, device=device).unsqueeze(0),
        }
    
    def get_action(self, state, deterministic=False):
        """
        返回 (action_scores_vec, action_idx, log_prob, value)
        action_scores_vec   : 发给 env.step() 的长度 = max_jobs 的一维 numpy
        action_idx          : 当前策略抽样 / argmax 得到的离散动作（任务下标）
        deterministic: 是否选择确定性动作（即概率最高的动作），在训练好以后的评估使用
        """
        with torch.no_grad():
            s = self.preprocess_state(state)                 # dict -> batch(1,…)
            logits = self.actor(s).squeeze(0)               # [max_jobs]
            probs  = F.softmax(logits, dim=-1)              # 有效任务上的分布

            if deterministic:
                action_idx = torch.argmax(probs)
                log_prob = torch.log(probs[action_idx] + 1e-10)  # 防止log(0)
            else:
                dist   = Categorical(probs)
                action_idx = dist.sample()                      # 采样动作
                log_prob = dist.log_prob(action_idx)            # 对应 log π(a|s)
            
            value    = self.critic(s).squeeze(0)

            # ② 构造发送给环境的分数向量：保证抽到的 idx 拥有最高分
            action_scores = torch.zeros_like(probs)
            action_scores[action_idx] = 1.0                 # one‑hot 即可
            # 如果想保留排序信息，可用
            # action_scores = probs.clone(); action_scores[action_idx] += 1e-3

        return action_scores.cpu().numpy(), action_idx.item(), log_prob.item(), value.item()
    
    def evaluate(self, num_episodes=10, model_path=None, deterministic=True):
        """
        评估训练好的策略
        Args:
            num_episodes: 评估的周期数
            model_path: 预训练模型的路径（如为None则使用当前网络）
            deterministic: 是否使用确定性策略（选择概率最高的动作）
        """
        # 加载预训练模型
        if model_path is not None:
            model_path = os.path.expanduser(model_path)
            self.actor.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded model from {model_path}")
        
        self.actor.eval()  # 设置评估模式
        
        # 初始化统计容器
        episode_rewards = []
        episode_jcts = []
        jct_means, jct_p95s, jct_p99s = [], [], []
        wait_means, wait_p95s, wait_p99s = [], [], []
        cost_means, cost_p95s, cost_p99s = [], [], []
        total_costs, budgets = [], []
        cost_budget_ratio = []

        for ep in range(num_episodes):
            state = self.env.reset(seed=0 + ep)  # 固定种子保证可复现
            done = False
            episode_reward = 0
            
            while not done:
                if self.env.visible_jobs:  # 有可见任务时选择动作
                    _, action_idx, _, _ = self.get_action(state, deterministic)
                    next_state, reward, done, _ = self.env.step(action_idx)
                else:  # 无任务时推进时间
                    next_state, reward, done, _ = self.env.step(-1)
                
                state = next_state
                episode_reward += reward
                done = self.env._is_done()  # 确认是否真正结束

            # 收集统计数据
            jct_vec ,jct_mean, jct_p95, jct_p99, wait_mean, wait_p95, wait_p99 = self.env.get_time_statistics()
            cost_mean, cost_p95, cost_p99, total_cost, budget = self.env.get_cost_statistics()

            episode_rewards.append(episode_reward)
            episode_jcts.append(jct_vec.copy())   # ★ 保存整条向量
            jct_means.append(jct_mean)
            jct_p95s.append(jct_p95)
            jct_p99s.append(jct_p99)
            wait_means.append(wait_mean)
            wait_p95s.append(wait_p95)
            wait_p99s.append(wait_p99)
            cost_means.append(cost_mean)
            cost_p95s.append(cost_p95)
            cost_p99s.append(cost_p99)
            total_costs.append(total_cost)
            budgets.append(budget)
            cost_budget_ratio.append(total_cost / budget)


            print(f"Episode {ep+1}/{num_episodes} | Reward: {episode_reward:.2f}")
            print(f"JCT - Mean: {jct_mean:.2f}h, P95: {jct_p95:.2f}h, P99: {jct_p99:.2f}h")
            print(f"Wait - Mean: {wait_mean:.2f}h, P95: {wait_p95:.2f}h, P99: {wait_p99:.2f}h")
            print(f"Cost - Total: {total_cost:.2f}, Budget: {budget:.2f}, Ratio: {total_cost / budget}\n")

        # 保存评估结果
        eval_dir = os.path.join(self.figures_path, "evaluation")
        os.makedirs(eval_dir, exist_ok=True)

        # ==== 按指标分文件保存 ====
        def save_metric(data, metric_name, columns, index_name="RL_Eval"):
            df = pd.DataFrame([data], columns=columns, index=[index_name])
            df.to_excel(os.path.join(eval_dir, f"{metric_name}_eval.xlsx"))

        # 1. JCT 统计
        save_metric(jct_means, "Mean_JCT_hours", list(range(1, num_episodes+1)))
        save_metric(jct_p95s, "P95_JCT_hours", list(range(1, num_episodes+1)))
        save_metric(jct_p99s, "P99_JCT_hours", list(range(1, num_episodes+1)))

        np.savez_compressed(
            os.path.join(eval_dir, "episode_jcts.npz"),
            episode_jcts=episode_jcts           # object array，每项可变长
        )

        # 2. 等待时间统计
        save_metric(wait_means, "Mean_Wait_Time_hours", list(range(1, num_episodes+1)))
        save_metric(wait_p95s, "P95_Wait_Time_hours", list(range(1, num_episodes+1)))
        save_metric(wait_p99s, "P99_Wait_Time_hours", list(range(1, num_episodes+1)))

        # 3. 费用统计
        save_metric(cost_means, "Mean_Cloud_Cost", list(range(1, num_episodes+1)))
        save_metric(cost_p95s, "P95_Cloud_Cost", list(range(1, num_episodes+1)))
        save_metric(cost_p99s, "P99_Cloud_Cost", list(range(1, num_episodes+1)))
        
        # 4. 总费用与预算
        pd.DataFrame([total_costs], columns=list(range(1, num_episodes+1)), 
                    index=["RL_Eval"]).to_excel(os.path.join(eval_dir, "Total_Cloud_Cost_eval.xlsx"))
        pd.DataFrame([[tc/b if b!=0 else 0 for tc, b in zip(total_costs, budgets)]], 
                    columns=list(range(1, num_episodes+1)), index=["RL_Eval"]
                    ).to_excel(os.path.join(eval_dir, "Cost_to_Budget_Ratio_eval.xlsx"))

        print(f"评估数据已保存至: {eval_dir}")


    
    def compute_gae(self, rewards, values, dones):
        """
        输入:
            rewards : list[float]         r_t
            values  : list[float]         V(s_t)  (长度 = len(rewards)+1)
            dones   : list[bool]          True 表示 s_{t+1} 是终止状态
        返回:
            torch.Tensor advantages  (len = len(rewards))
        """
        adv  = torch.zeros(len(rewards), device=self.device)   # 存优势 A_t
        last = 0.0                                             # 上一次递推的值
        for t in reversed(range(len(rewards))):                # 反向遍历
            mask  = 1.0 - dones[t]        # 到达终止状态时 mask = 0，截断未来回报
            delta = rewards[t] + self.gamma * values[t+1] * mask - values[t]
            # δ_t ＝ TD 误差   r_t + γV(s_{t+1}) − V(s_t)

            last  = delta + self.gamma * self.gae_lambda * mask * last
            # A_t = δ_t + γλ * A_{t+1}   （GAE 公式）

            adv[t] = last                # 记录当前 A_t
        return adv

    
    def compute_ranking_loss(self, pred_scores, target_orders):
        """计算排序损失"""
        batch_loss = 0
        for pred, target in zip(pred_scores, target_orders):
            valid_length = len(target)
            for i in range(valid_length):
                for j in range(i+1, valid_length):
                    # 确保目标中靠前的任务得分更高
                    batch_loss += torch.log(1 + 
                        torch.exp(pred[target[j]] - pred[target[i]]))
        return batch_loss
    
    def update(self):
        states, actions_idx, old_log_p, rewards, dones, values, is_dummy = zip(*self.buffer)
        
        values = list(values) + [0.0]  # 将最后一个 state 的 V 追加，便于 GAE

        # ---------------- 预处理全部状态一次 -------------
        jobs_b     = torch.stack([self.preprocess_state(s)['jobs'].squeeze(0)     for s in states])  # (B, J, 5)
        cluster_b  = torch.stack([self.preprocess_state(s)['cluster'].squeeze(0)  for s in states])  # (B, N)
        gpu_left_b = torch.stack([self.preprocess_state(s)['gpu_left'].squeeze(0) for s in states])  # (B, N, G)

        def slice_batch(idxs):
            return {
                'jobs'    : jobs_b[idxs],
                'cluster' : cluster_b[idxs],
                'gpu_left': gpu_left_b[idxs],
            }

        # ---------------- GAE -----------------------
        adv = self.compute_gae(rewards, values, dones)
        returns = adv + torch.tensor(values[:-1], device=self.device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # to tensor
        actions_idx = torch.tensor(actions_idx, device=self.device)
        old_log_p   = torch.tensor(old_log_p, device=self.device)
        is_dummy = torch.tensor(is_dummy, device=self.device, dtype=torch.bool)

        # 新增：把 -1 映射成 0，仅用于 log_prob 计算
        safe_actions = actions_idx.clone()
        safe_actions[is_dummy] = 0 

        # ---------------- PPO 迭代 ------------------
        idxs = np.arange(len(actions_idx))
        for _ in range(self.ppo_epochs):
            np.random.shuffle(idxs)
            for start in range(0, len(idxs), self.batch_size):
                mb = idxs[start:start + self.batch_size]

                s_mb = slice_batch(mb)
                logits = self.actor(s_mb)                          # (mb, max_jobs)
                dist   = Categorical(logits=logits)
                new_lp = dist.log_prob(safe_actions[mb])

                ratio  = torch.exp(new_lp - old_log_p[mb])
                surr1  = ratio * adv[mb]
                surr2  = torch.clamp(ratio, 1 - self.clip_epsilon,
                                            1 + self.clip_epsilon) * adv[mb]
                
                is_dummy_mb = is_dummy[mb]                     # ① 取本 batch 的掩码
                pg_term     = torch.min(surr1, surr2)

                entropy = dist.entropy().mean()        # (标量)

                if (~is_dummy_mb).any():                       # ② 防止全是 dummy
                    actor_loss = (
                        -pg_term[~is_dummy_mb].mean()  # PPO policy-gradient
                        - self.entropy_coef * entropy  # −α·H(π)
                    )
                else:
                    actor_loss = torch.tensor(0.0, device=self.device)

                v_pred = self.critic(s_mb).squeeze(-1)
                critic_loss = F.mse_loss(v_pred, returns[mb])

                loss = actor_loss + 0.5 * critic_loss

                self.actor_optim.zero_grad(); self.critic_optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.actor_optim.step(); self.critic_optim.step()

        self.buffer.clear()


    def plot_train_metrics(self, episode_rewards, window_size=30):
        fig_dir = self.figures_path
        os.makedirs(fig_dir, exist_ok=True)
        plt.figure(figsize=(15, 10))
    
        # 奖励曲线
        plt.subplot(2, 1, 1)
        plt.plot(episode_rewards, color='tab:blue', alpha=0.5, linewidth=2)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')

        # 滑动窗口平均奖励
        kernel = np.ones(window_size) / window_size
        moving_avg = np.convolve(episode_rewards, kernel, mode="valid")
        plt.subplot(2, 1, 2)
        plt.plot(range(window_size, len(episode_rewards) + 1),  # 横坐标往后平移 window_size‑1
         moving_avg,
         color="tab:orange",
         linewidth=2)
        plt.title(f"Moving‑Average Rewards (window={window_size})")
        plt.xlabel("Episode")
        plt.ylabel("Mean Reward")
    
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'training_metrics.png'))
        plt.close()

        # 保存数据到excel表格
        episode_numbers = list(range(1, len(episode_rewards) + 1))

        # Mean JCT
        df_rewards = pd.DataFrame([episode_rewards], columns=episode_numbers, index=['RL'])
        df_rewards.to_excel(os.path.join(fig_dir, 'Rewards.xlsx'))
    
    def plot_jct_statistics(self, episode_jct_mean, episode_jct_p95, episode_jct_p99):
        fig_dir = self.figures_path
        os.makedirs(fig_dir, exist_ok=True)
        plt.figure(figsize=(15, 10))

        # 第一个子图：Episode Mean JCT
        plt.subplot(3, 1, 1)
        plt.plot(episode_jct_mean, color='tab:blue', alpha=0.7, linewidth=2)
        plt.title('Episode Mean JCT')
        plt.xlabel('Episode')
        plt.ylabel('Mean JCT/hours')

        # 第二个子图：Episode P95 JCT
        plt.subplot(3, 1, 2)
        plt.plot(episode_jct_p95, color='tab:orange', alpha=0.7, linewidth=2)
        plt.title('Episode P95 JCT')
        plt.xlabel('Episode')
        plt.ylabel('P95 JCT/hours')

        # 第三个子图：Episode P99 JCT
        plt.subplot(3, 1, 3)
        plt.plot(episode_jct_p99, color='tab:red', alpha=0.7, linewidth=2)
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
        fig_dir = self.figures_path
        os.makedirs(fig_dir, exist_ok=True)
        plt.figure(figsize=(15, 10))

        # 第一个子图：Episode Mean Wait Time
        plt.subplot(3, 1, 1)
        plt.plot(episode_wait_time_mean, color='tab:blue', alpha=0.7, linewidth=2)
        plt.title('Episode Mean Wait Time')
        plt.xlabel('Episode')
        plt.ylabel('Mean Wait Time/hours')

        # 第二个子图：Episode P95 Wait Time
        plt.subplot(3, 1, 2)
        plt.plot(episode_wait_time_p95, color='tab:orange', alpha=0.7, linewidth=2)
        plt.title('Episode P95 Wait Time')
        plt.xlabel('Episode')
        plt.ylabel('P95 Wait Time/hours')

        # 第三个子图：Episode P99 Wait Time
        plt.subplot(3, 1, 3)
        plt.plot(episode_wait_time_p99, color='tab:red', alpha=0.7, linewidth=2)
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

    def plot_cloud_cost_statistics(self, episode_cloud_cost_mean, episode_cloud_cost_p95, episode_cloud_cost_p99, episode_total_cost, episode_budget):
        fig_dir = self.figures_path
        os.makedirs(fig_dir, exist_ok=True)
        plt.figure(figsize=(15, 12))

        # 图一
        # 第一个子图：Episode Mean Cloud Cost
        plt.subplot(3, 1, 1)
        plt.plot(episode_cloud_cost_mean, color='tab:blue', alpha=0.7, linewidth=2)
        plt.title('Episode Mean Cloud Cost')
        plt.xlabel('Episode')
        plt.ylabel('Mean Cloud Cost')

        # 第二个子图：Episode P95 Cloud Cost
        plt.subplot(3, 1, 2)
        plt.plot(episode_cloud_cost_p95, color='tab:orange', alpha=0.7, linewidth=2)
        plt.title('Episode P95 Cloud Cost')
        plt.xlabel('Episode')
        plt.ylabel('P95 Cloud Cost')

        # 第三个子图：Episode P99 Cloud Cost
        plt.subplot(3, 1, 3)
        plt.plot(episode_cloud_cost_p99, color='tab:red', alpha=0.7, linewidth=2)
        plt.title('Episode P99 Cloud Cost')
        plt.xlabel('Episode')
        plt.ylabel('P99 Cloud Cost')

        # 自动调整子图间距
        plt.tight_layout()

        # 保存图片
        plt.savefig(os.path.join(fig_dir, 'Cloud_Cost.png'))
        plt.close()

        # 图二
        plt.figure(figsize=(15, 12))
        
        # 第一个字图：Episode Total Cost
        plt.subplot(2, 1, 1)
        plt.plot(episode_total_cost, color='tab:blue', alpha=0.7, linewidth=2)
        plt.title('Episode Total Cost')
        plt.xlabel('Episode')
        plt.ylabel('Total Cost')
    
        # 第二个子图：Episode Total Cost to Budget Ratio
        ratio = [total / budget if budget != 0 else 0 for total, budget in zip(episode_total_cost, episode_budget)]
        plt.subplot(2, 1, 2)
        plt.plot(ratio, color='tab:green', alpha=0.7, linewidth=2)
        plt.title('Episode Total Cost / Budget Ratio')
        plt.xlabel('Episode')
        plt.ylabel('Cost to Budget Ratio')

        # 自动调整子图间距
        plt.tight_layout()

        # 保存图片
        plt.savefig(os.path.join(fig_dir, 'Total_Cloud_Cost.png'))
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

        # Cost to Budget Ratio
        df_ratio = pd.DataFrame([ratio], columns=episode_numbers, index=['RL'])
        df_ratio.to_excel(os.path.join(fig_dir, 'Cost_to_Budget_Ratio.xlsx'))

        # Total Cloud Cost
        df_total = pd.DataFrame([episode_total_cost], columns=episode_numbers, index=['RL'])
        df_total.to_excel(os.path.join(fig_dir, 'Total_Cloud_Cost.xlsx'))
    
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

    

    def collect_master_batch(self, batch_size=1024):
        """用 LCF 规则收集 (state, action) 对，返回张量 batch。"""
        obs_buf, act_buf = [], []
        env, prep = self.env, self.preprocess_state

        s = env.reset(seed=random.randint(0, 99999))
        while len(obs_buf) < batch_size:
            if env.visible_jobs:                                  # 有任务可选
                # -------- ① 先存当前 state --------------
                obs_buf.append(prep(s, to_gpu=False))             # 放 CPU，后面再搬 GPU

                # -------- ② 老师动作：LCF ---------------
                master_idx = min(range(len(env.visible_jobs)),
                            key=lambda i: env.visible_jobs[i].cost)
                act_buf.append(master_idx)                           # label

                # -------- ③ 环境前进一步 -----------------
                s, _, done, _ = env.step(master_idx)
            else:                                                 # 空时隙
                s, _, done, _ = env.step(-1)

            if done:                                              # 新 episode
                s = env.reset(seed=random.randint(0, 99999))

        # ---------- 拼 batch → GPU ----------
        obs_batch = {k: torch.cat([o[k] for o in obs_buf]).to(self.device)
                    for k in obs_buf[0]}
        act_batch = torch.tensor(act_buf, device=self.device, dtype=torch.long)
        return obs_batch, act_batch

    
    def pretrain_imitation(self,
                       steps=3000,
                       batch_size=1024,
                       ckpt_path="pretrain_actor_default_100steps_64bs.pth"):
        """纯监督训练 Actor，使其模仿专家策略。"""
        self.actor.train()
        opt = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        for step in range(1, steps + 1):
            s_batch, a_batch = self.collect_master_batch(batch_size)  # 得到启发式算法的数据集（状态和动作）

            logits = self.actor(s_batch)                 # 使用actor在当前状态下产生logits，形状(B, max_jobs)
            ce_loss = F.cross_entropy(logits, a_batch)   # 行为克隆
            ent_loss = Categorical(logits=logits).entropy().mean()
            loss = 0.9 * ce_loss - 0.1 * ent_loss        # +熵正则

            opt.zero_grad(); loss.backward(); opt.step()

            if step % 100 == 0:
                print(f"[BC] step {step}/{steps}, CE={ce_loss:.4f}")
                preds = logits.argmax(dim=-1)[:10].cpu().tolist()
                trues = a_batch[:10].cpu().tolist()
                print("pred:", preds, "label:", trues)

        torch.save(self.actor.state_dict(), ckpt_path)   # 保存权重
        self.actor.eval()


    
    def train(self, max_episodes=500):
        """训练主循环"""

        """
        # 打印GPU占用情况
        print(f"[GPU] Using device: {self.device}")
        print(f"[GPU] CUDA available: {torch.cuda.is_available()}")
        print(f"[GPU] Actor device: {next(self.actor.parameters()).device}")
        """
        
        episode_rewards = []
        #timesteps = 0

        # 统计数据
        episode_jct_mean = []
        episode_jct_p95 = []
        episode_jct_p99 = []

        episode_wait_time_mean = []
        episode_wait_time_p95 = []
        episode_wait_time_p99 = []
        episode_total_cost = []
        episode_budget = []

        episode_cloud_cost_mean = []
        episode_cloud_cost_p95 = []
        episode_cloud_cost_p99 = []

        result_list = []

        
        for episode in range(max_episodes):

            # 验证语句
            print(f"\n=== Starting Episode {episode+1} ===")
            print(f"Initial Visible Jobs: {len(self.env.visible_jobs)}")
            print(f"Total Jobs Remaining: {len(self.env.total_jobs)}")
            
            # 每个Episode都重置环境和初始化状态，也就是说每个Episode都是一整个调度过程（直到done）
            state = self.env.reset(seed = 2024 + episode)  # 会加载初始任务
            done = False
            episode_reward = 0
            episode_steps = 0
            
            while not done:
                if self.env.visible_jobs:                     # ① 有任务
                    _, action_idx, logp, value = self.get_action(state)
                    next_state, reward, done, _ = self.env.step(action_idx)
                    self.buffer.append((state, action_idx, logp, reward, done, value, False))
                else:                                         # ② 空时隙，相当于仅推进时隙
                    value = self.critic(self.preprocess_state(state)).item()
                    next_state, reward, done, _ = self.env.step(-1)
                    self.buffer.append((state, -1, 0.0, reward, done, value, True))
                    

                state           = next_state
                episode_reward += reward
                episode_steps += 1

                if len(self.buffer) >= self.buffer.maxlen:
                    self.update()
                
                done = self.env._is_done()  # 在外层循环外还得再调用一次，否则在visible_jobs为空，但还有任务没结束时，done无法正确结束
                
                # 验证语句，由于一个时隙会产生多次调度，因此有可能会重复打印多次相同的验证语句，不过实际仍在继续推进
                print(f"Finished: {len(self.env.finished_jobs)}/1000")
                print(f"Timeslot: {self.env.current_time}")
            # Episode 结束后把剩余样本也更新
            if self.buffer:
                self.update()

            # 记录统计数据
            _, jct_mean, jct_p95, jct_p99, wait_time_mean, wait_time_p95, wait_time_p99 = self.env.get_time_statistics()
            episode_jct_mean.append(jct_mean)
            episode_jct_p95.append(jct_p95)
            episode_jct_p99.append(jct_p99)

            episode_wait_time_mean.append(wait_time_mean)
            episode_wait_time_p95.append(wait_time_p95)
            episode_wait_time_p99.append(wait_time_p99)
            
            # 在费用方面
            cloud_cost_mean, cloud_cost_p95, cloud_cost_p99, total_cloud_cost, budget = self.env.get_cost_statistics()
            episode_cloud_cost_mean.append(cloud_cost_mean)
            episode_cloud_cost_p95.append(cloud_cost_p95)
            episode_cloud_cost_p99.append(cloud_cost_p99)
            episode_total_cost.append(total_cloud_cost)
            episode_budget.append(budget)

            # 记录帕累托前沿的数据
            log_row = {
                "method": 'RL',
                "param": self.env.budget_factor,
                "episode": episode,
                "cost": total_cloud_cost,
                "mean_jct": jct_mean,
                "mean_wait_time": wait_time_mean,
                "Cost to Budget Ratio": total_cloud_cost / budget
            }
            result_list.append(log_row)


            # 记录训练进度
            episode_reward += reward
            episode_rewards.append(episode_reward)
            print(
            f"Ep {episode+1:3d} | "
            f"Reward: {episode_reward:7.2f} | "
            )
            
            # 定期保存模型
            today = datetime.now().strftime("%Y-%m-%d")
            save_dir = os.path.expanduser(f"~/starburst/RLscheduler/checkpoints_default")  # ← 你的目标目录
            os.makedirs(save_dir, exist_ok=True)
            if (episode+1) % 50 == 0:
                torch.save(self.actor.state_dict(), os.path.join(save_dir, f"ppo_actor_{episode+1}.pth"))
                torch.save(self.critic.state_dict(), os.path.join(save_dir, f"ppo_critic_{episode+1}.pth"))
            
            # === episode 结束 ===
            self.actor_scheduler.step()
            self.critic_scheduler.step()

            # 同步衰减熵权重
            if (episode + 1) % self.lr_decay_step == 0:
                self.entropy_coef *= self.entropy_gamma
                print(f"[Decay] actor_lr={self.actor_scheduler.get_last_lr()[0]:.2e}, "
                    f"entropy_coef={self.entropy_coef:.4f}")
            
        
        # 训练结束后绘图
        self.plot_train_metrics(
            episode_rewards,
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
            episode_cloud_cost_p99,
            episode_total_cost,
            episode_budget
        )

        """
        df = pd.DataFrame(result_list)
        save_dir  = os.path.expanduser("~/starburst/RLscheduler/Pareto_curves/test_transformer")
        os.makedirs(save_dir, exist_ok=True)

        # ---------- 写 CSV ----------
        out_path  = os.path.join(save_dir, "all_runs.csv")
        if os.path.exists(out_path):
            df.to_csv(out_path, mode="a", header=False, index=False)
        else:
            df.to_csv(out_path, index=False)
        """
        
        
        return episode_rewards
    
    

# 使用示例
if __name__ == "__main__":
    # 创建环境
    num_nodes = 16
    episode=2000
    
    budget_factors_set = [0.01, 0.03, 0.05, 0.07, 0.1]
    num_nodes_set = [4, 8, 12, 16, 24]
    arrival_rate_set = [30, 45, 60, 90, 120]  # 平均每个时隙[10, 15, 20, 30, 40]
    wait_factor_set = [0.25, 0.75, 1.25, 2.5, 3]
    privacy_rate_set = [0.05, 0.1, 0.2, 0.4, 0.8]
    job_runtime_set = [0.5, 1, 2, 4, 8]  # 只有在synthetic下才可使用

    
    '''
    # 初始化PPO
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for arrival in arrival_rate_set:
        env = JobSchedulingEnv(num_nodes=num_nodes, arrival_rate=arrival)
        
        ppo = PPO(env, device=device)
        
        # 开始训练
        rewards = ppo.train(max_episodes=episode)
    '''
    
    
    
    
    # 初始化PPO
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = JobSchedulingEnv() 
    ppo = PPO(env, device=device)
    # -------- 行为克隆阶段 --------
    if not os.path.exists("pretrain_actor_default_2000steps_1024bs.pth"):
        ppo.pretrain_imitation(steps=2000, batch_size=1024, ckpt_path="pretrain_actor_default_2000steps_1024bs.pth")

    # -------- 载入权重，再 PPO --------
    ppo.actor.load_state_dict(torch.load("pretrain_actor_default_2000steps_1024bs.pth", map_location=device))
    # 开始训练

    #rewards = ppo.train(max_episodes=episode)

    # 在训练完成后调用评估
    ppo.evaluate(
        num_episodes=50,
        model_path="~/starburst/RLscheduler/checkpoints_default/ppo_actor_2000.pth",  # 只需要加载actor，下次记得保存好checkpoints，这个必须与figures的路径对应
        deterministic=True
    )
    
    

    