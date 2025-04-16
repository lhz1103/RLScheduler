import tf_agents
import tensorflow as tf
#from RLscheduler.job import Job


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


print(tf_agents.__version__)
