import tensorflow as tf
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import value_network

def create_actor_critic_networks(env):
    # 设置每层神经元数量
    actor_first_layer_nn = 64
    actor_sencond_layer_nn = 32
    critic_first_layer_nn = 64
    critic_sencond_layer_nn = 32
    # 定义Actor网络
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        env._state_spec(),
        env._action_spec(),
        fc_layer_params=(actor_first_layer_nn, actor_sencond_layer_nn),  # 可以根据需要调整网络结构
        activation_fn=tf.keras.activations.relu
    )
    
    # 为Actor网络添加批量归一化
    actor_net_layers = []
    for units in [actor_first_layer_nn, actor_sencond_layer_nn]:
        actor_net_layers.append(tf.keras.layers.Dense(units, activation=None))  # 去掉激活函数
        actor_net_layers.append(tf.keras.layers.BatchNormalization())  # 添加批量归一化
        actor_net_layers.append(tf.keras.layers.ReLU())  # 添加ReLU激活函数
    
    # 最后一层输出动作分布
    actor_net_layers.append(tf.keras.layers.Dense(env.action_spec().shape[0]))  # 输出动作空间维度
    actor_net = tf.keras.Sequential(actor_net_layers)
    
    # 定义Critic网络
    critic_net = value_network.ValueNetwork(
        env.observation_spec(),
        fc_layer_params=(critic_first_layer_nn, critic_sencond_layer_nn),  # 可以根据需要调整网络结构
        activation_fn=tf.keras.activations.relu
    )
    
    # 为Critic网络添加批量归一化
    critic_net_layers = []
    for units in [critic_first_layer_nn, critic_sencond_layer_nn]:
        critic_net_layers.append(tf.keras.layers.Dense(units, activation=None))  # 去掉激活函数
        critic_net_layers.append(tf.keras.layers.BatchNormalization())  # 添加批量归一化
        critic_net_layers.append(tf.keras.layers.ReLU())  # 添加ReLU激活函数
    
    # 最后一层输出状态价值
    critic_net_layers.append(tf.keras.layers.Dense(1))  # 输出一个标量，表示状态价值
    critic_net = tf.keras.Sequential(critic_net_layers)
    
    return actor_net, critic_net
