import copy
import numpy as np
from typing import Any, Dict, List, Optional

from tabulate import tabulate  # 数据转为表格格式
from tqdm import tqdm  # 显示进度条

from RLscheduler import Cluster, Job, utils, waiting_policy

DEFAULT_SIMULATOR_SPEC = {
    # Size of the cluster (i.e. # of cluster nodes).
    'cluster_size': 64,
    # Number of GPU(s) per cluster node.
    'gpus_per_node': 8,
    # Scheduling algorithm specifying order of the queue.
    'sched_alg': 'fifo',
    # How jobs are binpacked into the cluster.
    'binpack_alg': 'first-fit',
    # Waiting policy (how long jobs should wait in the cloud).
    'waiting_policy': 'linear_runtime',
    # Waiting hyperparameter (to be passed to waiting_policy)
    'waiting_factor': 1.25,
    # Sets clipping time for waiting (max time a job should wait)
    'clip_time': 1e9,
    # Enable backfill (assumes time estimator).
    'backfill': False,
    # Enable loop scheduling (just loop through entire queue, remove HoL).
    'loop': False,
    # Enable prediction. (Jobs predict if they can be assigned to cluster before timing out).
    # 0 is no prediction, 1 is perfect oracle
    'predict_wait': 0,
    # Queue length
    'max_queue_length': -1,
    # Long jobs wait
    'long_job_thres': -1,
    # Time estimator error
    'time_estimator_error': 0,
    # Data locality dealy
    'data_gravity': -1,
    # Pre-empt Cloud Ratio.
    # 2 means that the waiting policy is 2 * (waiting time), where waiting time is determined by waiting policy. If jobs times out with this waiting policy is it moved to the cloud.
    # If the job exceeds long_job_thres time on the cloud, it is moved back to the onprem.
    # The waiting time for onprem is now 2 * (waiting time) (for long jobs).
    'preempt_cloud_ratio': -1,
    # (Deprecated) Algorithm to immediately send job to cloud (without waiting).
    'filter_alg': None,
    # Prints out simulator state at every timestep.
    'verbose': False,
    # Appends python debugger at every timestemp.
    'debug': False,
    # Position for TQDM progress tracker bar.即进度条的位置
    'pbar_idx': 0,
    # Jobs to not consider for final metrics at the beg. and end. of simulator.热身任务，这些任务不会被考虑进最终统计指标
    'warmup_jobs': 5000,
    # Whether to get snapshots and save to result dict
    'snapshot': False,
    # Metadata on job generation (run prior to simulator).
    'jobgen_spec': {
        # Dataset type ['philly', 'philly_gen', 'gen_gpu']
        'dataset': 'philly',
        # Arrival rate of jobs (used in 'gen_gpu', 'philly_gen')
        'arrival_rate': -1,
        # Total number of jobs generated.
        'total_jobs': -1,
        # Avg. Job runtime (used in 'gen_gpu')
        'job_runtime': -1,
    },
}


def run_simulator(
        jobs: List[Job],
        simulator_spec: Optional[Dict[str, Any]] = DEFAULT_SIMULATOR_SPEC):
    """Executes a simulator over a fixed set of jobs. Returns a 
    a result dictionary over all finished jobs.

    Args:
        jobs: List of generated jobs sorted by their arrival times.
        simulator_spec: Simulator settings, see above dictionary for default
                        values.
    """
    _simulator_spec = DEFAULT_SIMULATOR_SPEC.copy()
    _simulator_spec.update(simulator_spec)  # 先复制默认配置，再使用用户提供的simulator_spec来更新它，这样可避免修改原始的默认值，同时将用户传入的设置融合到模拟器的配置中。
    simulator_spec = _simulator_spec

    # TODO(mluo): convert into class fields instead of manual indexing.
    sched_alg = simulator_spec['sched_alg']
    sort_func = utils.generate_sorting_function(sched_alg)  # 根据设置里的调度算法名称生成排序函数

    waiting_policy_str = simulator_spec['waiting_policy'].split('-')  # 用 - 分割waiting_policy的字段
    assert len(waiting_policy_str) <= 2  #  确保等待策略不超过两个部分
    if len(waiting_policy_str) == 2:
        simulator_spec['waiting_policy'] = waiting_policy_str[0]  # 第一个字段是策略名称
        simulator_spec['waiting_factor'] = float(waiting_policy_str[1])  # 第二个是参数

    waiting_fn = waiting_policy.lookup_linear_function(  # 使用该函数查找当前设置匹配的等待策略
        simulator_spec['waiting_policy'],
        waiting_factor=simulator_spec['waiting_factor'])
    binpack_alg = simulator_spec['binpack_alg']
    backfill = simulator_spec['backfill']
    loop = simulator_spec['loop']
    predict_wait = simulator_spec['predict_wait']
    clip_time = simulator_spec['clip_time']
    filter_alg = simulator_spec['filter_alg']
    max_queue_length = simulator_spec['max_queue_length']
    time_estimator_error = simulator_spec['time_estimator_error'] / 100.0
    long_job_thres = simulator_spec['long_job_thres']
    preempt_cloud_ratio = simulator_spec['preempt_cloud_ratio']
    data_gravity = simulator_spec['data_gravity']
    if preempt_cloud_ratio >0:
        assert long_job_thres > 0, 'Must set long_job_thres > 0 if preempt_cloud_ratio > 0'
    assert not (
        backfill and loop
    ), f'Must only set one option to be True - backfill:{backfill}, loop:{loop} '  # 两者互斥，只能选择其中一个

    verbose = simulator_spec['verbose']
    debug = simulator_spec['debug']
    snapshot = simulator_spec['snapshot']
    # Initialize simulator variables\
    jobs = copy.deepcopy(jobs)
    queue = []
    finished_jobs = []
    num_jobs = len(jobs)
    cloud_cost = 0.0
    # Create fake cluster. The cluster is homogeneous.
    cluster = Cluster(num_nodes=simulator_spec['cluster_size'],
                      num_gpus_per_node=simulator_spec['gpus_per_node'],
                      backfill=backfill,
                      binpack=binpack_alg)
    t = 0
    pbar = tqdm(total=len(jobs),
                desc="Jobs progress: ",
                position=simulator_spec['pbar_idx'])

    snapshots = {}
    total_cloud_jobs = 0
    # Simulation Loop - Continues until all jobs have passed and the queue is empty and the cluster has no more jobs.
    # 即循环成立在：job列表中有未到达的任务 或 queue中有等待的任务 或 cluster.active_jobs有正在运行的任务
    while len(jobs) > 0 or len(queue) > 0 or cluster.active_jobs:
        # Clear cluster of jobs that have completed
        completed_jobs = cluster.try_clear(t)
        finished_jobs.extend(completed_jobs)  # 列表（list）的一种方法，它用于将一个可迭代对象（比如另一个列表、元组、字符串等）中的元素逐个添加到原列表的末尾

        '''
        以下三个小循环的功能：
        1. 在等待队列中遍历每个任务，检查是否需要上云；
        2. 遍历任务列表中的所有任务，查看是否有任务到达，有的话就加入等待队列，并设置等待时间上限；
        3. 先把等待队列按照排序算法（FIFO，SJF等）进行排序，再遍历等待队列，查看目前任务能否放入集群中运行。
        '''
        # Check for jobs that have waited too long (move to cloud).
        i = 0
        while i < len(queue):  # 在队列中遍历
            job = queue[i]
            # If job has timed out, send to cloud.
            if t > job.deadline - job.runtime:  # 超过最晚开始时间
                raise ValueError(
                    f'Job {job.idx} has timed out: {t} > {job.deadline}')
            elif t == job.deadline - job.runtime:  # 已经是最晚开始时间，仍在排队，就上传到云端
                queue.remove(job)
                job.state = 'TIMEOUT-CLOUD'
                # Shortcut: Job can predict it will go to cloud or not, if so, it would have began running at job.arrival.
                # Perfect Oracle
                if predict_wait == 1:  # 设置在云端的任务开始时间
                    job.start = job.arrival
                elif predict_wait == 0 or predict_wait == 2:
                    job.start = job.deadline - job.runtime
                else:
                    raise ValueError(
                        f'Predict wait {predict_wait} wrong value!')
                
                if preempt_cloud_ratio > 0:
                    # If a job has not been prempted to the cloud before.如果任务未被抢占
                    if not job.preempt_cloud:
                        # Emulate preemption from cloud back to onprem.任务运行时间>阈值，则说明是长任务，还需返回集群
                        if job.runtime > long_job_thres:
                            # Job will run on the cloud for long_job_thres and then move back to onprem.
                            job.preempt_cloud = True
                            # The job will arrive again at this time (original arrival + original waiting time + long_job_thres)
                            job.new_arrival = job.deadline - job.runtime + long_job_thres  # 更新到达时间
                            job.start = None
                            job.deadline = None
                            job.state = None
                            
                            # Here, we add it back into the arrival jobs, sorted by arrival time.根据新到达时间重新将任务排入列表中
                            # Finding the correct position for the new object
                            position = 0
                            for obj in jobs:
                                # Design Choice: Should we insert by its original arrival or new arrival?
                                # Here, we insert by the new arrival.
                                if obj.arrival > job.new_arrival:
                                    break
                                position += 1
                            # Inserting the object at the found position
                            jobs.insert(position, job)
                            # Cloud cost incurred includes the time job ran for LONG_JOB_THRES on the cloud.
                            cloud_cost += job.cost / (job.runtime/long_job_thres)  # 调整云端费用，即阈值*GPU数量
                            pbar.update(-1) # 更新进度条
                            continue  # 继续下一轮循环
                cloud_cost += job.cost
                total_cloud_jobs += 1
                finished_jobs.append(job)
            else:  # 当前任务没超时，继续检查队列的下一个任务
                i += 1

        # Add jobs to queue that have arrived. Jobs are assumed to have been ordered by arrival times.
        i = 0
        while i < len(jobs):
            job = jobs[i]
            if job.arrival < t and job.new_arrival < t:
                raise ValueError("Should not have entered here!")
            elif job.arrival == t or job.new_arrival==t:
                jobs.remove(job)  # 从列表移除该任务，说明它进入等待队列或开始处理
                deadline = waiting_fn(job)  # 根据等待策略设置等待时间的上限
                if job.preempt_cloud:
                    arrival = job.new_arrival
                else:
                    arrival = job.arrival

                if preempt_cloud_ratio > 0:
                    # Mutate waiting function.
                    waiting_time = max(0.0, deadline - job.runtime - arrival)
                    if job.preempt_cloud:  # star-wait的逻辑：没上过云的直接上云，上过云的等待时间设置为preempt_cloud_ratio*原始等待时间
                        # Star-Wait applies a laonger waiting policy for cloud preempted jobs.
                        waiting_time = preempt_cloud_ratio * waiting_time
                    else:
                        # Star-Wait originally applies No-Wait policy
                        waiting_time = 0 
                    deadline = arrival + job.runtime + waiting_time
                job.set_deadline(deadline)  # 设置最终等待时间的上限
                
                if deadline == -1 or (preempt_cloud_ratio <0 and job.runtime < long_job_thres):   
                    # For Constant-Wait + No-SJ (job offloading)，针对固定等待+直接将小任务（job.runtime < long_job_thres）上云
                    job.state = 'TIMEOUT-CLOUD'
                    job.start = arrival
                    job.set_deadline(deadline=arrival + job.runtime)
                    cloud_cost += job.cost
                    finished_jobs.append(job)
                else:
                    # For time estimator ablations.时间误差（文章里似乎没有这个）
                    if time_estimator_error != 0:  
                        original_runtime = job.runtime
                        mod_runtime = original_runtime + np.random.normal(
                            loc=0.0,
                            scale=time_estimator_error * original_runtime)  # 给运行时间加上一个随机扰动
                        mod_runtime = max(0, mod_runtime)
                        if 'filter' in simulator_spec['waiting_policy']:  # 在waiting_policy中包含filter的话，则进一步调整deadline
                            deadline = arrival + simulator_spec[
                                'waiting_factor'] * (
                                    job.resources['GPUs'] +
                                    job.resources['CPUs'] /
                                    53.0) * mod_runtime + job.runtime
                    waiting_time = max(0.0,
                                       deadline - job.runtime - arrival)
                    if waiting_time < 0:
                        raise ValueError('Waiting time should not be negative.')
                    waiting_time = min(clip_time, waiting_time)
                    job.set_deadline(deadline=arrival + job.runtime +
                                     waiting_time)
                    queue.append(job)
                pbar.update(1)
            else:
                break

        # Sort the queue based on the queueing order algorithm. (FIFO, SJF, etc.)
        queue.sort(key=sort_func)

        # Go through queue and fit jobs onto cluster as needed
        i = 0
        while i < len(queue):  # 遍历等待队列，看该任务目前能否放入集群中
            job = queue[i]
            can_fit, _ = cluster.try_fit_v2(t, job)
            if can_fit:  # 如果集群可以运行该任务
                queue.remove(job)
                #queue.extend(preempted_jobs)
            elif not loop:  # 如果没有超序调度，只能检查队列第一个任务，
                break
            else:  # 如果任务无法调度，但loop为true（超序调度？）就继续遍历下一个
                i += 1

        # Perform EASY backfilling (assumes time estimator).
        if backfill:
            # Reserve the first element of queue that is blocking
            if queue:
                job_to_reserve = queue[0]
                # Reserving large jobs for backfilling (like in Slurm).为队列头部的任务预留资源
                can_reserve = cluster.try_reserve(t, job_to_reserve)
                # If can't reserve within reasonble time, leave the job in the queue.
                if not can_reserve:
                    pass  
                else:
                    queue.remove(job_to_reserve)  # 如果可以预留资源，则把该任务移出队列，防止它阻塞后续的任务
                i = 0
                while i < len(queue):
                    job = queue[i]
                    can_fit, preempted_jobs = cluster.try_fit_v2(t, job)
                    if can_fit:
                        queue.remove(job)
                        #queue.extend(preempted_jobs)
                    else:
                        i += 1

        if max_queue_length != -1:  # 设置了最长队列限制
            while len(queue) > max_queue_length:
                q_job = queue[-1]  # 从末尾取出任务并移除
                queue.remove(q_job)
                q_job.state = 'TIMEOUT-CLOUD'  # 直接上传到云端
                q_job.start = t
                if q_job.preempt_cloud:  # 如果已经在云端运行过又返回集群的
                    q_job.set_deadline(deadline=q_job.new_arrival +
                                       q_job.runtime)
                else:
                    q_job.set_deadline(deadline=q_job.arrival + q_job.runtime)
                cloud_cost += q_job.cost
                finished_jobs.append(q_job)

        if snapshot:
            if t not in snapshots:
                snapshots[t] = {}
            snapshots[t]['new_queue'] = copy.deepcopy(queue)  # 记录当前队列状态

        # Skip to next timestep (matches algorithm 1 in paper). The next timestep is the minimum of（确定下一个时间步）:
        # 1) a new job either arrives (first elmeent in job queue)：新任务到达
        # 2) job finishes on the cluster：集群中任务完成
        # 3) existing job in the queue times out.：队列中任务超时
            
        # Case 2
        next_time_list = []  # 存储所有可能的下一个时间步
        for _, job in cluster.active_jobs.items():
            next_time_list.append(job.start + job.runtime)  # 第二种情况下，任务完成时间是下一个时间步的候选值

        # Case 1
        if len(jobs) > 0:
            cur_job = jobs[0]
            if cur_job.preempt_cloud:
                next_time_list.append(cur_job.new_arrival)
            else:
                next_time_list.append(cur_job.arrival)  # 第一种情况下，新任务到来的时间是下一个时间步的候选值
        
        # Case 3
        if queue:
            for q in queue:
                # append time outs
                next_time_list.append(q.deadline - q.runtime)  # 第三种情况下，任务超时的时间是下一个时间步的候选值

        # If there are no jobs left in the cluster and in the job and queue, terminate simulation.
        if len(next_time_list) == 0:  # 如果 next_time_list 为空，意味着没有更多的作业需要处理，仿真应结束。此时检查队列和待处理作业是否都为空，如果不为空则报错，确保仿真正确结束
            assert len(queue) == 0 and len(jobs) == 0
            break

        if min(next_time_list) < t and abs(min(next_time_list) - t) > 1e-6:  # 确保下一个时间步的时间不会比当前时间回退
            print(simulator_spec)
            raise ValueError(
                f'Simulator cannot go back in time, there is a bug: {t}->{min(next_time_list)}'
            )
        t = min(next_time_list)  # 推进仿真时间，进入下一个时间步
        if verbose or debug:  # 打印仿真状态
            headers = [
                'Timestamp', 'Cloud Cost', 'Queue Length', 'Jobs Left',
                'Finished Jobs'
            ]
            data = [(t, cloud_cost, len(queue), len(jobs), len(finished_jobs))]
            print(tabulate(data, headers=headers))
            if debug:
                import pdb
                pdb.set_trace()

    end_sim_jobs = cluster.try_clear(1e12)  # 仿真结束时的检查
    assert len(end_sim_jobs) == 0 and len(jobs) == 0 and len(
        queue) == 0, 'Simulator did not finish properly. There are still running jobs in the cluster.'
    
    # Sort jobs by their initial arrival (aka idx).
    finished_jobs.sort(key=lambda x: x.idx)

    # Generate final logs for the simulator.
    result_dict = {
        'idx': np.array([j.idx for j in finished_jobs]),
        'arrival': np.array([j.arrival for j in finished_jobs]),
        'start': np.array([j.start for j in finished_jobs]),
        'runtime': np.array([j.runtime for j in finished_jobs]),
        'deadline': np.array([j.deadline for j in finished_jobs]),
        'num_gpus': np.array([j.num_gpus for j in finished_jobs]),
        'state': np.array([j.state for j in finished_jobs]),
        'allocated_gpus': np.array([j.allocated_gpus for j in finished_jobs]),
        'simulator_spec': simulator_spec,
        'stats': {}
    }

    if snapshot:
        result_dict['snapshot'] = snapshots
    # Computing Simulator stats, such as avg. waiting, avg. JCT, cloud cost, utilization.（计算仿真统计数据）
    total_waiting_time = 0.0
    total_running_time = 0.0
    num_jobs = 0
    total_cloud_cost = 0
    sum_local_space = 0.0
    sum_cloud_space = 0.0

    start_time = finished_jobs[simulator_spec['warmup_jobs']].arrival  # 仿真开始与结束时间（排除了热身任务）
    end_time = finished_jobs[len(finished_jobs) -
                             simulator_spec['warmup_jobs'] - 1].arrival

    jct_list = []
    wait_list = []
    if data_gravity!=-1:  # 如果设置了数据重力，则需要额外添加云端由数据重力时间带来的费用和运行时间
        for job in finished_jobs:
            if job.state == 'TIMEOUT-CLOUD':
                job.cost = job.cost * (1 + data_gravity/job.runtime)
                job.runtime = job.runtime + data_gravity
    for job in finished_jobs:  # 遍历所有已完成的任务
        inter_start = max(job.start, start_time)  # inter_start和inter_end是为了裁减掉热身期的占用时间，确保统计时不考虑热身期的资源占用
        inter_end = min(job.start + job.runtime, end_time)
        # Cut off beginning and ending of simulator to reach steady state. Calculate the "bleeding".
        if job.idx < simulator_spec['warmup_jobs'] or job.idx > len(
                finished_jobs) - simulator_spec['warmup_jobs']:  # 计算热身任务的情况
            if job.state == 'LOCAL':
                if inter_end >= inter_start:
                    sum_local_space += job.num_gpus * (inter_end - inter_start)
            elif job.state == 'TIMEOUT-CLOUD':
                if inter_end >= inter_start:
                    sum_cloud_space += job.num_gpus * (inter_end - inter_start)
            continue
        # Moved to cloud
        if job.state == 'TIMEOUT-CLOUD':
            total_waiting_time += job.start - job.arrival
            if inter_end >= inter_start:
                sum_cloud_space += job.num_gpus * (inter_end - inter_start)
            total_cloud_cost += job.cost
        elif job.state == 'LOCAL':
            total_waiting_time += job.start - job.arrival
            if inter_end >= inter_start:
                sum_local_space += job.num_gpus * (inter_end - inter_start)
            if job.preempt_cloud:
                total_cloud_cost += job.cost / (job.runtime/long_job_thres)
        else:
            raise ValueError(f'Job {job.idx} has invalid state: {job.state}')
        jct_list.append(job.runtime + job.start - job.arrival)
        wait_list.append(job.start - job.arrival)
        total_running_time += job.runtime
        num_jobs += 1

    result_dict['stats']['total_cloud_cost'] = total_cloud_cost
    result_dict['stats']['avg_cloud_cost'] = total_cloud_cost / (end_time -
                                                                 start_time)  # 对时间取平均
    result_dict['stats']['avg_waiting'] = total_waiting_time / num_jobs
    result_dict['stats']['avg_jct'] = (total_waiting_time +
                                       total_running_time) / num_jobs
    result_dict['stats']['90_jct'] = np.percentile(jct_list,
                                                   90,
                                                   method='nearest')
    result_dict['stats']['99_jct'] = np.percentile(jct_list,
                                                   99,
                                                   method='nearest')

    result_dict['stats']['avg_wait'] = np.mean(wait_list)
    result_dict['stats']['90_wait'] = np.percentile(wait_list,
                                                    90,
                                                    method='nearest')
    result_dict['stats']['99_wait'] = min(
        24, np.percentile(wait_list, 99, method='nearest'))

    result_dict['stats']['cluster_utilization'] = sum_local_space / (
        simulator_spec['cluster_size'] * simulator_spec['gpus_per_node'] *
        (end_time - start_time))
    result_dict['stats']['system_utilization'] = (
        sum_local_space + sum_cloud_space) / (simulator_spec['cluster_size'] *
                                              simulator_spec['gpus_per_node'] *
                                              (end_time - start_time))

    stats_dict = result_dict['stats']
    headers = [
        'Sched Policy', 'Waiting Policy', '# Cluster Nodes',
        'Total Cloud Cost', 'Avg. Cloud Cost', 'Avg. Waiting', 'Avg. JCT',
        '90th JCT', '99th JCT', 'Cluster Utilization', 'System Utilization'
    ]
    waiting_policy_str = simulator_spec['waiting_policy']
    waiting_factor_str = simulator_spec['waiting_factor']
    data = [(simulator_spec['sched_alg'], \
        f'{waiting_policy_str}-{waiting_factor_str}', simulator_spec['cluster_size'], \
        stats_dict['total_cloud_cost'], stats_dict['avg_cloud_cost'], \
        stats_dict['avg_waiting'], stats_dict['avg_jct'], stats_dict['90_jct'], stats_dict['99_jct'], stats_dict['cluster_utilization'], stats_dict['system_utilization'])]
    print(tabulate(data, headers=headers))
    return result_dict
