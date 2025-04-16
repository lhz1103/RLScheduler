import tensorflow as tf
import models
import JobSchedulingEnv
from tf_agents.agents.ppo import ppo_agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from tf_agents.environments import tf_py_environment
from tf_agents.trajectories import trajectory
import job_generator


# 创建环境
dataset_config = {
        'dataset': 'philly_gen',
        'arrival_rate': 8,
        'cv_factor': 1.0,
        'total_jobs': 1000,
        'seed': 2024,
        'privacy_rate': 0.2,
        'job_runtime': 4.0
        }
jobs = job_generator.load_processed_jobs(dataset_config)
env = JobSchedulingEnv(total_jobs = jobs, num_nodes=3, num_gpus_per_node=8)
train_env = tf_py_environment.TFPyEnvironment(env)

# 创建Actor-Critic网络
actor_net, critic_net = models.create_actor_critic_networks(train_env)

# 创建PPO Agent
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
agent = ppo_agent.PPOAgent(
    time_step_spec=train_env.time_step_spec(),
    action_spec=train_env.action_spec(),
    optimizer=optimizer,
    actor_net=actor_net,
    value_net=critic_net,
    entropy_regularization=0.01,
    importance_ratio_clipping=0.2,
    normalize_observations=False,
    normalize_rewards=False
)

agent.initialize()

# 经验回放缓存（PPO需要轨迹数据）
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=1000
)

# 数据收集函数（需要完整轨迹）
def collect_trajectories(env, policy, buffer, num_episodes=10):
    for _ in range(num_episodes):
        time_step = env.reset()
        policy_state = policy.get_initial_state(env.batch_size)
        
        while not time_step.is_last():
            action_step = policy.action(time_step, policy_state)
            next_time_step = env.step(action_step.action)
            # 打印奖励和观测信息
            print("当前观测:", time_step.observation.numpy())
            print("获得奖励:", next_time_step.reward.numpy())
            traj = trajectory.from_transition(time_step, action_step, next_time_step)
            buffer.add_batch(traj)
            time_step = next_time_step

# 训练循环
num_iterations = 1000
batch_size = 32

for iter in range(num_iterations):
    # 收集轨迹数据
    collect_trajectories(train_env, agent.collect_policy, replay_buffer)
    
    # 从缓存采样轨迹
    dataset = replay_buffer.as_dataset(
        sample_batch_size=batch_size,
        num_steps=2
    ).prefetch(3)
    
    # 训练PPO Agent
    iterator = iter(dataset)
    for _ in range(10):  # 每个迭代训练10次
        experience, _ = next(iterator)
        train_loss = agent.train(experience).loss