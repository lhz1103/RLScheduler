import Queue
import time
import numpy as np
import parameters as pm
from cluster import Cluster
import log
from scheduler_base import Scheduler


class RL_Env(Scheduler):
    def __init__(self, name, trace, logger, training_mode=True):
        Scheduler.__init__(self, name, trace, logger)

        self.epsilon = 0.0  # 用于表示探索的概率，通常在强化学习中用于 epsilon-greedy 策略（当前为 0.0，意味着完全依赖于当前的策略而不进行探索）
        self.training_mode = training_mode
        self.sched_seq = []  # 表示已经被调度的任务
        self.job_prog_in_ts = dict()  # 用于存储每个任务在每个时间步（timeslot）的进度
        self.window_jobs = None  # 当前窗口（时隙）内的任务
        self.jobstats = dict()  # 统计信息
        for stats_name in [
                "arrival", "ts_completed", "tot_completed", "duration",
                "uncompleted", "running", "total", "backlog", "cpu_util",
                "gpu_util"
        ]:
            self.jobstats[stats_name] = []
        if pm.PS_WORKER and pm.BUNDLE_ACTION:  # PS_WORKER如果为false则可以只考虑worker（GPU）
            self.action_freq = [0 for _ in range(3)]  # 只有三个动作：worker、ps、bundle
        # prepare for the first timeslot
        self._prepare()

    def _prepare(self):  # 为每个时隙做好准备
        # admit new jobs
        num_arrv_jobs = 0  # 当前时隙到达的任务数量
        if self.curr_ts in self.trace:  # 如果当前时隙在trace中存在
            for job in self.trace[self.curr_ts]:  # 从trace中获取当前时隙的任务并加入到未完成的任务集合中
                job.reset()
                self.uncompleted_jobs.add(job)
                if not self.training_mode:
                    job.training = False
                num_arrv_jobs += 1
                self.logger.debug(job.info())
        self.jobstats["arrival"].append(num_arrv_jobs)  # 记录当前时隙到达的任务数
        self.jobstats["total"].append(
            len(self.completed_jobs) + len(self.uncompleted_jobs))  # 记录当前已经完成的任务数和未完成的任务数之和（总任务数）
        self.jobstats["backlog"].append(
            max(len(self.uncompleted_jobs) - pm.SCHED_WINDOW_SIZE, 0))  # 记录任务积压数（未完成的任务数-窗口大小（时隙内最大调度的任务数？））

        # reset
        self._sched_states()  # get scheduling states in this ts
        self.running_jobs.clear()  # 清空正在运行的任务？
        self.node_used_resr_queue = Queue.PriorityQueue()  
        for i in range(pm.CLUSTER_NUM_NODES):
            self.node_used_resr_queue.put((0, i))  # 按资源使用情况排序集群中的节点
        self.cluster.clear()  # 清空集群？

        for job in self.uncompleted_jobs:
            if pm.ASSIGN_BUNDLE and pm.PS_WORKER:  # assign each job a bundle of ps and worker first to avoid job starvation，但是我的场景下不能这么用
                _, node = self.node_used_resr_queue.get()  # 取出一个空闲节点
                resr_reqs = job.resr_worker + job.resr_ps
                succ, node_used_resrs = self.cluster.alloc(resr_reqs, node)  # 尝试分配资源
                if succ:  # 如果分配成功
                    job.num_ps = 1
                    job.curr_ps_placement = [node]
                    job.num_workers = 1
                    job.curr_worker_placement = [node]
                    job.dom_share = np.max(1.0 *
                                           (job.num_workers * job.resr_worker +
                                            job.num_ps * job.resr_ps) /
                                           self.cluster.CLUSTER_RESR_CAPS)
                    self.running_jobs.add(job)  # 将任务添加到正在运行的任务列表中
                else:
                    job.num_workers = 0
                    job.curr_worker_placement = []
                    job.num_ps = 0
                    job.curr_ps_placement = []
                    job.dom_share = 0
                self.node_used_resr_queue.put(
                    (np.sum(node_used_resrs),
                     node))  # always put back to avoid blocking in step()，把资源再放回节点
            else:
                job.num_workers = 0
                job.curr_worker_placement = []
                if pm.PS_WORKER:
                    job.num_ps = 0
                    job.curr_ps_placement = []
                job.dom_share = 0

        if pm.VARYING_SKIP_NUM_WORKERS:
            self.skip_num_workers = np.random.randint(1, pm.MAX_NUM_WORKERS)
        else:
            self.skip_num_workers = 8  #np.random.randint(0,pm.MAX_NUM_WORKERS)
        if pm.VARYING_PS_WORKER_RATIO:
            self.ps_worker_ratio = np.random.randint(3, 8)
        else:
            self.ps_worker_ratio = 5

    def _move(self):  # 推进到下一个时隙
        self._progress()
        if len(self.completed_jobs) == pm.TOT_NUM_JOBS:  # 总体时间上，所有的任务都完成，设置环境结束
            self.end = True
        else:
            # next timeslot
            self.curr_ts += 1  # 推进到下一个时隙，在超过最大时间步长时抛出异常
            if self.curr_ts > pm.MAX_TS_LEN:
                self.logger.error(
                    "Exceed the maximal number of timeslot for one trace!")
                self.logger.error("Results: " + str(self.get_results()))
                self.logger.error("Stats: " + str(self.get_jobstats()))
                for job in self.uncompleted_jobs:
                    self.logger.error("Uncompleted job " + str(job.id) +
                                      " tot_epoch: " + str(job.num_epochs) +
                                      " prog: " + str(job.progress) +
                                      " workers: " + str(job.num_workers))
                raise RuntimeError
            self._prepare()

    # step forward by one action
    def step(self, output):
        # mask and adjust probability
        mask = np.ones(pm.ACTION_DIM)  # 全1的mask数组，大小是动作空间的维度（我的动作空间维度应该如何设置？），如果值值为0表示屏蔽该动作
        for i in range(len(self.window_jobs)):  # 遍历当前时隙内的所有任务
            if self.window_jobs[  # 该位置没有任务时，屏蔽相关的动作
                    i] is None:  # what if job workers are already maximum
                if pm.PS_WORKER:
                    if pm.BUNDLE_ACTION:  # 开启BUNDLE_ACTION有三个动作，需进行屏蔽 worker, ps, bundle
                        mask[3 * i] = 0.0
                        mask[3 * i + 1] = 0.0
                        mask[3 * i + 2] = 0.0
                    else:  # 没开BUNDLE_ACTION有2个动作
                        mask[2 * i] = 0.0
                        mask[2 * i + 1] = 0.0
                else:
                    mask[i] = 0.0
            else:  # 有任务时
                if pm.PS_WORKER:
                    worker_full = False
                    ps_full = False
                    if self.window_jobs[i].num_workers >= pm.MAX_NUM_WORKERS:  # 记录当前任务的工作节点数已达到最大限制
                        worker_full = True
                    if self.window_jobs[i].num_ps >= pm.MAX_NUM_WORKERS:
                        ps_full = True
                    if worker_full:  # 相应屏蔽对应的动作
                        if pm.BUNDLE_ACTION:
                            mask[3 * i] = 0.0
                        else:
                            mask[2 * i] = 0.0
                    if ps_full:
                        if pm.BUNDLE_ACTION:
                            mask[3 * i + 1] = 0.0
                        else:
                            mask[2 * i + 1] = 0.0
                    if (worker_full or ps_full) and pm.BUNDLE_ACTION:
                        mask[3 * i + 2] = 0.0

        masked_output = np.reshape(output[0] * mask, (1, len(mask)))  # 对模型的输出进行屏蔽，确保无效动作的概率为 0，其中output[0]是每个动作的概率
        sum_prob = np.sum(masked_output)  # 计算有效动作的总概率
        action_vec = np.zeros(len(mask))  # 初始化一个与动作空间大小相同的零向量，用来记录所选动作的索引
        move_on = True
        valid_state = False
        if ((not pm.PS_WORKER) and sum(mask[:len(self.window_jobs)]) == 0) \
          or (pm.PS_WORKER and (not pm.BUNDLE_ACTION) and sum(mask[:2*len(self.window_jobs)]) == 0) \
          or (pm.PS_WORKER and pm.BUNDLE_ACTION and sum(mask[:3*len(self.window_jobs)]) == 0):  # 没有任务可选择
            self.logger.debug(
                "All jobs are None, move on and do not save it as a sample")
            self._move()
        elif sum_prob <= 0:  # 所有动作都被屏蔽或有概率为 1 的动作被屏蔽，表示出现异常情况
            self.logger.info(
                "All actions are masked or some action with probability 1 is masked!!!"
            )
            if pm.EXPERIMENT_NAME is None:
                self.logger.info(
                    "Output: " + str(output)
                )  # Output: [[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0. 0.  1.  0.]], WHY?
                self.logger.info("Mask: " + str(mask))
                self.logger.info("Window_jobs: " + str(self.window_jobs))
                num_worker_ps_str = ""
                for job in self.window_jobs:
                    if job:
                        num_worker_ps_str += str(job.id) + ": " + str(
                            job.num_ps) + " " + str(job.num_workers) + ","
                self.logger.info("Job: " + num_worker_ps_str)  # 记录当前输出、屏蔽信息以及任务信息
            self._move()  # 跳过当前时隙
        else:  # 正常情况
            masked_output = masked_output / sum_prob  # 归一化屏蔽后的概率，使有效动作概率和为1
            if self.training_mode:  # 根据概率选择动作，训练模式
                # select action
                if np.random.rand(
                ) > pm.MASK_PROB:  # only valid for training mode，根据 MASK_PROB 设置，决定是否使用模型输出的概率进行动作选择，还是根据原始的 output[0]
                    masked_output = np.reshape(output[0], (1, len(mask)))
                action_cumsum = np.cumsum(masked_output)  # 计算累积概率
                action = (action_cumsum > np.random.randint(1, pm.RAND_RANGE) /
                          float(pm.RAND_RANGE)).argmax()  # 根据累积概率选择一个动作，模拟随机探索

                if pm.EPSILON_GREEDY: 
                    if np.random.rand() < self.epsilon:  # 以 epsilon 的概率随机选择一个动作，避免陷入局部最优解
                        val_actions = []
                        for i in range(len(masked_output[0])):
                            if masked_output[0][
                                    i] > pm.MIN_ACTION_PROB_FOR_SKIP:
                                val_actions.append(i)
                        action = val_actions[np.random.randint(
                            0, len(val_actions))]  # 从所有有效动作中随机选择一个

                if pm.INJECT_SAMPLES:  # 在模型训练时，基于资源利用情况、任务的状态以及配置的概率，决定是否强制跳过某些动作或选择特定动作，避免陷入局部最优或保证资源分配平衡
                    if (not pm.REAL_SPEED_TRACE) and (not pm.PS_WORKER):
                        allMaxResr = True  # 所有任务资源使用情况都已经达到最大，即没有可以跳过的任务
                        for job in self.window_jobs:
                            if job:
                                if job.num_workers > self.skip_num_workers:  # 检查当前任务的worker是否大于允许跳过的资源数量
                                    continue
                                else:
                                    allMaxResr = False
                                    break
                        if allMaxResr and masked_output[0][len(  # 如果所有作业的资源都已达到最大，并且当前动作是跳过时间步（通常是len(action_vec) - 1），
                                action_vec  # 且该动作的概率大于 MIN_ACTION_PROB_FOR_SKIP（最小跳过概率），可以考虑注入跳过动作
                        ) - 1] > pm.MIN_ACTION_PROB_FOR_SKIP and np.random.rand(  # 进一步检查一个随机数是否小于等于 SAMPLE_INJECTION_PROB，即根据设定的概率决定是否真的选择跳过
                        ) <= pm.SAMPLE_INJECTION_PROB:  # choose to skip if prob larger than a small num, else NaN
                            action = len(action_vec) - 1  # 如果满足以上条件，强制选择跳过动作
                            self.logger.debug("Got 1.")
                    elif pm.REAL_SPEED_TRACE and pm.PS_WORKER:
                        # shuffle = np.random.choice(len(self.window_jobs), len(self.window_jobs), replace=False)  # shuffle is a must, otherwise NN selects only the first several actions!!!
                        if pm.JOB_RESR_BALANCE and pm.BUNDLE_ACTION:
                            max_num_ps_worker = 0
                            min_num_ps_worker = 10**10
                            index_min_job = -1
                            for i in range(len(self.window_jobs)):  # 遍历所有任务
                                job = self.window_jobs[i]
                                if job:
                                    num_ps_worker = job.num_ps + job.num_workers  # 记录每个任务的需求资源总和
                                    if num_ps_worker > max_num_ps_worker:
                                        max_num_ps_worker = num_ps_worker
                                    if num_ps_worker < min_num_ps_worker:
                                        min_num_ps_worker = num_ps_worker
                                        index_min_job = i  # 记录最大值和最小值以及最小需求资源的任务索引
                            if min_num_ps_worker and index_min_job != -1 and max_num_ps_worker / min_num_ps_worker > np.random.randint(
                                    3, 6):  # 只有当存在 资源最少的作业 且 资源差距 大于3-6辈的某个随机值时（说明存在资源不平衡的情况），才会考虑强制选择动作
                                if masked_output[0][
                                        3 * index_min_job +
                                        2] > pm.MIN_ACTION_PROB_FOR_SKIP and masked_output[
                                            0][3 *
                                               index_min_job] > pm.MIN_ACTION_PROB_FOR_SKIP:  # 资源不平衡的动作概率大于MIN_ACTION_PROB_FOR_SKIP才考虑选择该动作
                                    if np.random.rand() < 0.5:
                                        action = 3 * index_min_job + 2  # 选择bundle动作
                                    else:
                                        action = 3 * index_min_job  # 选择worker动作

                        shuffle = [_ for _ in range(len(self.window_jobs))]  # 初始化包含任务顺序的列表
                        for i in shuffle:
                            job = self.window_jobs[i]
                            if job:
                                if pm.BUNDLE_ACTION:
                                    # if one of three actions: ps/worker/bundle has low probability, enforce to select it
                                    if min(self.action_freq) > 0 and min(
                                            self.action_freq) * 1.0 / sum(
                                                self.action_freq) < 0.001:  # 如果某个动作的选择频率极低（低于总频率的 0.1%）
                                        index = np.argmin(self.action_freq)  # 选择频率最小的动作
                                        if mask[3 * i +
                                                index] > 0 and masked_output[0][
                                                    3 * i +
                                                    index] > pm.MIN_ACTION_PROB_FOR_SKIP:  # 如果该动作的掩码（mask）值大于 0 且该动作的概率值（masked_output）大于阈值 MIN_ACTION_PROB_FOR_SKIP，则强制选择该动作
                                            action = 3 * i + index
                                            self.logger.debug("Got 0: " +
                                                              str(index))
                                            break
                                    if (job.num_workers == 0
                                            or job.num_ps == 0):  # 如果任务没有worker或ps需求
                                        if job.num_ps == 0 and job.num_workers == 0 and mask[
                                                3 * i +
                                                2] > 0 and masked_output[0][
                                                    3 * i +
                                                    2] > pm.MIN_ACTION_PROB_FOR_SKIP and np.random.rand(
                                                    ) < 0.5:
                                            action = 3 * i + 2  # 如果任务既没有 PS 也没有 Worker 且 mask[3 * i + 2] > 0 且该动作的概率值大于 MIN_ACTION_PROB_FOR_SKIP，则以 50% 的概率选择 bundle 动作
                                            self.logger.debug("Got 1")
                                        if job.num_workers == 0 and mask[
                                                3 *
                                                i] > 0 and masked_output[0][
                                                    3 *
                                                    i] > pm.MIN_ACTION_PROB_FOR_SKIP:
                                            action = 3 * i  # 如果任务没有worker，选择worker动作
                                        if job.num_ps == 0 and mask[
                                                3 * i +
                                                1] > 0 and masked_output[0][
                                                    3 *
                                                    i] > pm.MIN_ACTION_PROB_FOR_SKIP:
                                            action = 3 * i + 1  # 如果任务没有ps，选择ps动作
                                        break
                                    elif job.num_ps > job.num_workers * self.ps_worker_ratio and np.random.rand(  # 当 PS 任务的数量大于 Worker 数量的某个比例时，说明该作业的 PS 任务可能过多
                                    ) < 0.5:
                                        if mask[3 * i + 2] > 0 and masked_output[0][
                                                3 * i +
                                                2] > pm.MIN_ACTION_PROB_FOR_SKIP and mask[
                                                    3 *
                                                    i] > 0 and masked_output[0][
                                                        3 *
                                                        i] > pm.MIN_ACTION_PROB_FOR_SKIP:
                                            if np.random.rand() < 0.5:
                                                # increase this job's bundle
                                                action = 3 * i + 2
                                                self.logger.debug("Got 2.")
                                            else:
                                                action = 3 * i
                                                self.logger.debug("Got 2.")
                                            break
                                    elif job.num_workers >= job.num_ps * 0.5 and np.random.rand(  # 当 Worker 任务的数量大于 PS 任务的数量的 50% 时
                                    ) < 0.5:
                                        if mask[3 * i + 2] > 0 and masked_output[0][
                                                3 * i +
                                                2] > pm.MIN_ACTION_PROB_FOR_SKIP and mask[
                                                    3 * i +
                                                    1] > 0 and masked_output[0][
                                                        3 * i +
                                                        1] > pm.MIN_ACTION_PROB_FOR_SKIP:
                                            if np.random.rand() < 0.01:
                                                # increase this job's bundle
                                                action = 3 * i + 2
                                                self.logger.debug("Got 3.")
                                            else:
                                                # incrase ps
                                                action = 3 * i + 1
                                                self.logger.debug("Got 4.")
                                            break
                                else:  # 未启用BUNDLE_ACTION时
                                    if job.num_workers == 0 and mask[  # 如果没有worker需求
                                            2 * i] > 0 and masked_output[0][
                                                2 *
                                                i] > pm.MIN_ACTION_PROB_FOR_SKIP and np.random.rand(
                                                ) < 0.01:
                                        action = 2 * i  # 以极小的概率选择worker
                                        self.logger.debug("Got 1.")
                                        break
                                    elif job.num_ps == 0 and mask[
                                            2 * i +
                                            1] > 0 and masked_output[0][
                                                2 * i +
                                                1] > pm.MIN_ACTION_PROB_FOR_SKIP and np.random.rand(
                                                ) < 0.01:
                                        action = 2 * i + 1
                                        self.logger.debug("Got 2.")
                                        break
                                    elif job.num_ps >= job.num_workers * self.ps_worker_ratio and mask[
                                            2 * i] > 0 and masked_output[0][
                                                2 *
                                                i] > pm.MIN_ACTION_PROB_FOR_SKIP and np.random.rand(
                                                ) < 0.5:
                                        # increase this job's worker
                                        action = 2 * i
                                        self.logger.debug("Got 3.")
                                        break
                                    elif job.num_workers >= job.num_ps * self.ps_worker_ratio and mask[
                                            2 * i +
                                            1] > 0 and masked_output[0][
                                                2 * i +
                                                1] > pm.MIN_ACTION_PROB_FOR_SKIP and np.random.rand(
                                                ) < 0.5:
                                        # increase this job's ps
                                        action = 2 * i + 1
                                        self.logger.debug("Got 4.")
                                        break
            else:  # 验证模式
                if pm.SELECT_ACTION_MAX_PROB:  # only available for validation，控制是否在验证模式下始终选择最大概率的动作
                    action = np.argmax(
                        masked_output
                    )  # output is [[...]] # always select the action with max probability
                else:  # 对 masked_output 进行累积求和
                    action_cumsum = np.cumsum(masked_output)
                    action = (action_cumsum >
                              np.random.randint(1, pm.RAND_RANGE) /
                              float(pm.RAND_RANGE)).argmax()  # 选择第一个概率累积大于随机阈值的动作索引，通过在累积概率分布上 采样 来进行动作选择

            action_vec[action] = 1  # 在action_vec中记录选择的动作
            # check whether skip this timeslot
            if pm.SKIP_TS and action == len(action_vec) - 1:
                self._move()  # 移动到下一个时隙
                # filter out the first action that causes 0 reward??? NO
                # if sum([job.num_workers+job.num_ps for job in self.uncompleted_jobs]) > 0:
                valid_state = True
                self.sched_seq.append(None)  # 没有添加调度任务
                self.logger.debug("Skip action is selected!")
                self.logger.debug("Output: " + str(output))
                self.logger.debug("Masked output: " + str(masked_output))
            else:
                # count action freq
                if pm.PS_WORKER and pm.BUNDLE_ACTION:
                    self.action_freq[action % 3] += 1  # 更新动作频率

                # allocate resource
                if pm.PS_WORKER:
                    if pm.BUNDLE_ACTION:
                        job = self.window_jobs[action / 3]  # 得到动作对应的任务
                    else:
                        job = self.window_jobs[action / 2]
                else:
                    job = self.window_jobs[action]
                if job is None:
                    self._move()
                    self.logger.debug("The selected action is None!")
                else:
                    _, node = self.node_used_resr_queue.get()  # 获取一个将要分配资源的计算节点
                    # get resource requirement of the selected action
                    if pm.PS_WORKER:
                        if pm.BUNDLE_ACTION:
                            if action % 3 == 0:
                                resr_reqs = job.resr_worker
                            elif action % 3 == 1:
                                resr_reqs = job.resr_ps
                            else:
                                resr_reqs = job.resr_worker + job.resr_ps
                        else:
                            if action % 2 == 0:  # worker
                                resr_reqs = job.resr_worker
                            else:
                                resr_reqs = job.resr_ps
                    else:
                        resr_reqs = job.resr_worker
                    succ, node_used_resrs = self.cluster.alloc(resr_reqs, node)  # 进行资源分配
                    if succ:  # 如果分配成功
                        move_on = False
                        # change job tasks and placement
                        if pm.PS_WORKER:
                            if pm.BUNDLE_ACTION:
                                if action % 3 == 0:  # worker
                                    job.num_workers += 1
                                    job.curr_worker_placement.append(node)
                                elif action % 3 == 1:  # ps
                                    job.num_ps += 1
                                    job.curr_ps_placement.append(node)
                                else:  # bundle
                                    job.num_ps += 1
                                    job.curr_ps_placement.append(node)
                                    job.num_workers += 1
                                    job.curr_worker_placement.append(node)
                            else:
                                if action % 2 == 0:  # worker
                                    job.num_workers += 1
                                    job.curr_worker_placement.append(node)
                                else:  # ps
                                    job.num_ps += 1
                                    job.curr_ps_placement.append(node)
                        else:
                            job.num_workers += 1
                            job.curr_worker_placement.append(node)

                        job.dom_share = np.max(
                            1.0 * (job.num_workers * job.resr_worker +
                                   job.num_ps * job.resr_ps) /
                            self.cluster.CLUSTER_RESR_CAPS)  # 任务的资源份额，根据任务当前的 worker 和 ps 数量 和 集群容量 计算
                        self.node_used_resr_queue.put(
                            (np.sum(node_used_resrs), node))  # 更新节点的资源使用情况
                        self.running_jobs.add(job)  # 记录当前任务正在运行
                        valid_state = True
                        self.sched_seq.append(job)  # 记录当前任务已经被调度
                    else:  # 资源分配失败时
                        self._move()
                        self.logger.debug("No enough resources!")
        if move_on:  # 是否可以继续执行？，根据这个设置奖励
            reward = self.rewards[-1] * move_on
        else:
            reward = 0
        return masked_output, action_vec, reward, move_on, valid_state  # invalid state, action and output when move on except for skip ts

    def get_jobstats(self):
        self.jobstats["duration"] = [(job.end_time - job.arrv_time + 1)
                                     for job in self.completed_jobs]  # 计算了每个已完成作业的持续时间，并将其存储在 self.jobstats 字典中
        for name, value in self.jobstats.items():
            self.logger.debug(name + ": length " + str(len(value)) + " " +
                              str(value))  # 打印每个统计信息的名称、长度和具体值
        return self.jobstats

    def _sched_states(self):
        self.states = []
        for job in self.running_jobs:
            self.states.append((job.id, job.type, job.num_workers, job.num_ps))  # 记录当前所有正在运行的作业的基本信息

    def get_job_reward(self):
        job_reward = []
        for job in self.sched_seq:  # 遍历已经调度任务序列
            if job is None:  # skip，跳过的任务
                if len(self.job_prog_in_ts) > 0:  # 如果 job_prog_in_ts 中有元素，表示该作业已经有进度
                    job_reward.append(self.rewards[-1] /
                                      len(self.job_prog_in_ts))  # 当前奖励除以该时间片中作业进度的数量
                else:
                    job_reward.append(0)  # 没有进度则奖励为0
            else:
                job_reward.append(self.job_prog_in_ts[job])  # 不为None则获取该任务进度并作为奖励
        self.sched_seq = []
        self.job_prog_in_ts.clear()  # 清空已经调度序列和进度

        self.logger.info("Action Frequency: " + str(self.action_freq))
        return job_reward

    def get_sched_states(self):
        return self.states  # 返回当前调度状态，参考_sched_states函数

    def _progress(self):  # 进度更新
        reward = 0  # 时隙内的总奖励
        num_ts_completed = 0  # 初始化已完成的作业数量为 0
        for job in self.running_jobs:
            norm_prog = job.step() / job.num_epochs
            self.job_prog_in_ts[job] = norm_prog  # 每个作业通过调用 job.step() 更新它的进度，并将其规范化为进度比例存储在字典中
            reward += norm_prog  # 奖励就是归一化的进度
            if job.progress >= job.real_num_epochs:  # 作业进度超过实际的 real_num_epochs（即完成任务）
                if pm.FINE_GRAIN_JCT:  # 以下是计算结束时间
                    job.end_time = self.curr_ts - 1 + job.get_run_time_in_ts()  # 当前的时隙-1+任务在本时隙中运行的时间 
                else:
                    job.end_time = self.curr_ts
                # self.running_jobs.remove(job) # it means running in this ts, so no need to delete
                self.uncompleted_jobs.remove(job)
                self.completed_jobs.add(job)
                num_ts_completed += 1
        self.rewards.append(reward)

        self.jobstats["running"].append(len(self.running_jobs))  # 更新统计信息，添加本时隙内的信息
        self.jobstats["tot_completed"].append(len(self.completed_jobs))
        self.jobstats["uncompleted"].append(len(self.uncompleted_jobs))
        self.jobstats["ts_completed"].append(num_ts_completed)
        cpu_util, gpu_util = self.cluster.get_cluster_util()
        self.jobstats["cpu_util"].append(cpu_util)
        self.jobstats["gpu_util"].append(gpu_util)


def test():
    import log, trace
    logger = log.getLogger(name="agent_" + str(id), level="INFO")
    job_trace = trace.Trace(logger).get_trace()
    env = RL_Env("RL", job_trace, logger)
    while not env.end:
        data = env.step()
        for item in data:
            print (item)
        print ("-----------------------------")
        input("Next? ")

    print (env.get_results())


if __name__ == '__main__':
    test()
