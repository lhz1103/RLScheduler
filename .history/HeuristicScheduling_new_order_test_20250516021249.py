import numpy as np
from collections import deque
import random
from job import Job
from job_generator import load_processed_jobs
import matplotlib.pyplot as plt
import os
import pandas as pd

"""
5.15：
1. 运行synthetic_ex的实验
"""

class heuristic:
    def __init__(self, env, strategy = 'FIFO'):  
        
        self.env = env
        self.schedule_policy = strategy
        self.figures_path = f"~/starburst/RLscheduler/test_synthetic_ex_rate = 0.4" 

    def get_action(self):
        """根据策略生成的排序队列，返回队列第一个任务
            FIFO：时间顺序
            SJF：占用GPU时间少的任务优先
            SmallGPUsFirst：占用GPU少的任务优先
            BigGPUsFirst：占用GPU大的任务优先
            LongestJobFirst：占用GPU时间长的任务优先
            LongestWaitingTimeFirst：等待时间长的任务优先
            HighestCostFirst：费用大的任务优先
            LowestCostFirst:费用小的任务优先
            待实现
            ：各属性加权
        """

        # 根据策略排序任务
        if self.schedule_policy == 'FIFO':
            sorted_jobs = self.env.visible_jobs  # 默认FIFO顺序
        elif self.schedule_policy == 'SJF':
            sorted_jobs = sorted(self.env.visible_jobs, key=lambda x: x.runtime)
        elif self.schedule_policy == 'SmallGPUsFirst':
            sorted_jobs = sorted(self.env.visible_jobs, key=lambda x: x.num_gpus)
        elif self.schedule_policy == 'BigGPUsFirst':
            sorted_jobs = sorted(self.env.visible_jobs, key=lambda x: x.num_gpus, reverse=True)
        elif self.schedule_policy == 'LongestJobFirst':
            sorted_jobs = sorted(self.env.visible_jobs, key=lambda x: x.runtime, reverse=True) 
        elif self.schedule_policy == 'LowestCostFirst':  
            sorted_jobs = sorted(self.env.visible_jobs, key=lambda x: x.cost)
        elif self.schedule_policy == 'HighestCostFirst':  
            sorted_jobs = sorted(self.env.visible_jobs, key=lambda x: x.cost, reverse=True)
        elif self.schedule_policy == 'LongestWaitingTimeFirst':
            sorted_jobs = sorted(self.env.visible_jobs, key=lambda x: x.waitingtime, reverse=True)
        elif self.schedule_policy == 'OutOfOrder':
            gpu_free = self.env.cluster_state.copy()                 # (N,) 可用 GPU
            for job in self.env.visible_jobs:
                placement = self.env._try_local_place(job, gpu_free)
                if placement is not None:                            # 能放下
                    return self.env.visible_jobs.index(job)          # 直接用原索引
            # 如果一个都放不下，则退化为 FIFO
            sorted_jobs = self.env.visible_jobs
        else:
            raise ValueError(f"未知策略: {self.schedule_policy}")

        return self.env.visible_jobs.index(sorted_jobs[0])
    
    def append_method_to_excel(self, file_path, method_name, data_row):
        '''往excel里添加其他方法的数据'''
        file_path = os.path.expanduser(file_path)
        # Episode 列表自动从数据长度生成
        episode_numbers = list(range(1, len(data_row) + 1))
    
        # 创建新的 DataFrame 行
        new_row_df = pd.DataFrame([data_row], columns=episode_numbers, index=[method_name])
    
        if os.path.exists(file_path):
            # 文件存在，读取原始文件
            existing_df = pd.read_excel(file_path, index_col=0)

            # 如果已有相同 method_name 的行，则删除原行
            if method_name in existing_df.index:
                existing_df = existing_df.drop(method_name)
        
            # 合并原始数据和新数据
            updated_df = pd.concat([existing_df, new_row_df])
        else:
            # 文件不存在，直接使用新数据
            updated_df = new_row_df

        # 写回 Excel，替换为更新后的版本
        updated_df.to_excel(file_path)
    
    def log_env_state(self, episode: int, timeslot: int):
        """
        将当前环境状态写入日志文件。
        参数:
            episode: 当前 episode 序号（从1开始）
            timeslot: 当前时隙（通常为 self.env.current_time 的值）
        """
        # 确保日志目录存在
        log_dir = os.path.expanduser("~/starburst/RLscheduler/log")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    
        # 日志文件名为 self.schedule_policy.txt
        log_file = os.path.join(log_dir, f"{self.schedule_policy}.txt")
    
        # 提取每个队列的简单信息（此处只记录任务的 idx，可根据需要扩展）
        queue_ids = [job.idx for job in self.env.queue]
        active_ids = [job.idx for job in self.env.active_jobs]
        visible_ids = [job.idx for job in self.env.visible_jobs]
    
        # 写入日志：可以根据具体需求记录更多信息（比如任务的 arrival、runtime、需求等）
        log_str = (f"Episode {episode}, Timeslot {timeslot}:\n"
               f"  Active Jobs: {self.env.active_jobs}\n"
               f"  Visible Jobs: {self.env.visible_jobs}\n"
               f"  Cluster State: {self.env.cluster_state.tolist()}\n"
               f"  Finished Jobs: {self.env.finished_jobs}\n\n")
    
        with open(log_file, "a") as f:
            f.write(log_str)


    
    def train(self, max_episodes=1000, update_interval=2048):
        """训练主循环"""
        episode_rewards = []

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
        episode_total_cost = []
        episode_budget = []

        result_list = []

        episode_actions_list = []  # 用于存储每个 episode 的 a_idx 序列


        
        for episode in range(max_episodes):

            # 验证语句
            print(f"\n=== Starting Episode {episode+1} ===")
            print(f"Initial Visible Jobs: {len(self.env.visible_jobs)}")
            print(f"Total Jobs Remaining: {len(self.env.total_jobs)}")
            
            # 每个Episode都重置环境和初始化状态，也就是说每个Episode都是一整个调度过程（直到done）
            state = self.env.reset(seed = 13 + episode)
            done = False
            episode_reward = 0
            episode_steps = 0
            
            while not done:
                if self.env.visible_jobs:                      # ① 调度一个任务
                    a_idx = self.get_action()
                    
                else:                                          # ② 推进到下一个时隙
                    a_idx = -1

                next_state, reward, done, _ = self.env.step(a_idx)

                if episode == len(episode_actions_list):
                    episode_actions_list.append([])  # 初始化本 episode 的动作列表
                episode_actions_list[-1].append(a_idx)


                episode_reward += reward
                state = next_state
                
                print(f"Timeslot: {self.env.current_time}")
                
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
            
            cloud_cost_mean, cloud_cost_p95, cloud_cost_p99, total_cloud_cost, budget = self.env.get_cost_statistics()
            episode_cloud_cost_mean.append(cloud_cost_mean)
            episode_cloud_cost_p95.append(cloud_cost_p95)
            episode_cloud_cost_p99.append(cloud_cost_p99)
            episode_total_cost.append(total_cloud_cost)
            episode_budget.append(budget)


            """
            # 保存每个 episode 的 a_idx 和指标
            episode_log_dir = os.path.expanduser("~/starburst/RLscheduler/log/episode_actions")
            os.makedirs(episode_log_dir, exist_ok=True)

            episode_log_path = os.path.join(episode_log_dir, f"{self.schedule_policy}_actions_log.csv")

            # 创建 DataFrame 行
            row_data = {
                "episode": episode + 1,
                "actions": str(episode_actions_list[-1]),
                "mean_jct": jct_mean,
                "mean_wait_time": wait_time_mean,
                "total_cloud_cost": total_cloud_cost
            }

            if not os.path.exists(episode_log_path):
                # 首次写入带表头
                pd.DataFrame([row_data]).to_csv(episode_log_path, mode='w', header=True, index=False)
            else:
                # 追加写入
                pd.DataFrame([row_data]).to_csv(episode_log_path, mode='a', header=False, index=False)
            """
            

            # 记录帕累托前沿，更改费用预算
            log_row = {
                "method": self.schedule_policy,
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
            
            
        # 必须存在这些数据文件才能使用
        # 训练结束保存数据，追加到excel中
        self.append_method_to_excel(f"{self.figures_path}/Rewards.xlsx", method_name= self.schedule_policy, data_row = episode_rewards)

        self.append_method_to_excel(f"{self.figures_path}/Mean_JCT_hours.xlsx", method_name= self.schedule_policy, data_row = episode_jct_mean)
        self.append_method_to_excel(f"{self.figures_path}/P95_JCT_hours.xlsx", method_name= self.schedule_policy, data_row = episode_jct_p95)
        self.append_method_to_excel(f"{self.figures_path}/P99_JCT_hours.xlsx", method_name= self.schedule_policy, data_row = episode_jct_p99)

        self.append_method_to_excel(f"{self.figures_path}/Mean_Wait_Time_hours.xlsx", method_name= self.schedule_policy, data_row = episode_wait_time_mean)
        self.append_method_to_excel(f"{self.figures_path}/P95_Wait_Time_hours.xlsx", method_name= self.schedule_policy, data_row = episode_wait_time_p95)
        self.append_method_to_excel(f"{self.figures_path}/P99_Wait_Time_hours.xlsx", method_name= self.schedule_policy, data_row = episode_wait_time_p99)

        self.append_method_to_excel(f"{self.figures_path}/Mean_Cloud_Cost.xlsx", method_name= self.schedule_policy, data_row = episode_cloud_cost_mean)
        self.append_method_to_excel(f"{self.figures_path}/P95_Cloud_Cost.xlsx", method_name= self.schedule_policy, data_row = episode_cloud_cost_p95)
        self.append_method_to_excel(f"{self.figures_path}/P99_Cloud_Cost.xlsx", method_name= self.schedule_policy, data_row = episode_cloud_cost_p99)

        self.append_method_to_excel(f"{self.figures_path}/Total_Cloud_Cost.xlsx", method_name= self.schedule_policy, data_row = episode_total_cost)
        ratio = [total / budget if budget != 0 else 0 for total, budget in zip(episode_total_cost, episode_budget)]
        self.append_method_to_excel(f"{self.figures_path}/Cost_to_Budget_Ratio.xlsx", method_name= self.schedule_policy, data_row = ratio)
        # 写入csv文件
        df = pd.DataFrame(result_list)
        csv_dir = os.path.expanduser("~/starburst/RLscheduler/Pareto_curves")
        os.makedirs(csv_dir, exist_ok=True)
        csv_path = os.path.join(csv_dir, "all_runs.csv")

        # 追加写；若文件不存在则带表头写入
        header_needed = not os.path.exists(csv_path)
        df.to_csv(csv_path, mode="a", header=header_needed, index=False)
        
        return episode_rewards
    
    

# 本次启发式算法使用的是linearCost等待，因为在环境中设置没改
if __name__ == "__main__":
    # 假设已实现Job类和JobSchedulingEnv
    from HeuristicEnv_new_order import JobSchedulingEnv
    
    # 创建环境
    num_nodes = 16
    episodes = 500
    
    env = JobSchedulingEnv()

    FIFO = heuristic(env, strategy = 'FIFO')
    rewards_FIFO = FIFO.train(max_episodes=episodes)

    
    SJF = heuristic(env, strategy = 'SJF')
    rewards_SJF = SJF.train(max_episodes=episodes)

    SGF = heuristic(env, strategy = 'SmallGPUsFirst')
    rewards_SGF = SGF.train(max_episodes=episodes)


    LCF = heuristic(env, strategy = 'LowestCostFirst')
    rewards_LCF = LCF.train(max_episodes=episodes)
 
    





