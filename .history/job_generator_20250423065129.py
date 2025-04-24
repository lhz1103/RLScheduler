import numpy as np
from typing import Any, Dict, List

from job import Job
import waiting_policy
from traces import philly
from traces import helios


# Returns the total cost of a GPU-only job.
def gpu_cost_fn(num_gpus: int, runtime: float):
    return num_gpus * runtime
# 这里把cost = 资源 * 时间

def load_processed_jobs(dataset_config: Dict[str, Any]):  # 从配置中读取数据集类型（如 philly 或 helios等），调用对应的函数加载和处理任务数据。
    # arrival_rate指每小时到达任务数，
    dataset_type = dataset_config['dataset']
    if dataset_type == 'philly':
        philly_jobs = philly.load_philly_traces('~/philly-traces/trace-data')
        jobs = process_philly_jobs(philly_jobs)
        get_timeslot(jobs)
        return jobs
    elif dataset_type == 'philly_gen':
        philly_jobs = philly.load_philly_traces('~/philly-traces/trace-data')
        dataset_kwargs = {
            'total_jobs': dataset_config['total_jobs'],
            'arrival_rate': dataset_config['arrival_rate'],
            'cv_factor': dataset_config['cv_factor'],
            'seed': dataset_config['seed'],
            'privacy_rate': dataset_config['privacy_rate'],
        }
        jobs = generate_philly_gpu_jobs(philly_jobs, **dataset_kwargs)
        get_timeslot(jobs)
        return jobs
    elif dataset_type == 'gen_gpu':
        philly_jobs = philly.load_philly_traces('~/philly-traces/trace-data')
        dataset_kwargs = {
            'total_jobs': dataset_config['total_jobs'],
            'arrival_rate': dataset_config['arrival_rate'],
            'job_runtime': dataset_config['job_runtime'],
            'seed': dataset_config['seed'],
            'privacy_rate': dataset_config['privacy_rate'],
        } 
        jobs = generate_gpu_jobs(philly_jobs, **dataset_kwargs)
        get_timeslot(jobs)
        return jobs
    elif dataset_type == 'helios':
        helios_jobs = helios.load_helios_traces('~/HeliosData/data/Venus')
        jobs = process_helios_jobs(helios_jobs)
        get_timeslot(jobs)
        return jobs
    elif dataset_type == 'helios_gen':
        helios_jobs = helios.load_helios_traces('~/HeliosData/data/Venus')
        dataset_kwargs = {
            'total_jobs': dataset_config['total_jobs'],
            'arrival_rate': dataset_config['arrival_rate'],
            'cv_factor': dataset_config['cv_factor'],
            'seed': dataset_config['seed'],
            'privacy_rate': dataset_config['privacy_rate'],
        }
        jobs = generate_helios_jobs(helios_jobs, **dataset_kwargs)
        get_timeslot(jobs)
        return jobs
    elif dataset_type == 'synthetic':
        dataset_kwargs = {
            'total_jobs': dataset_config['total_jobs'],
            'arrival_rate': dataset_config['arrival_rate'],
            'job_runtime': dataset_config['job_runtime'],
            'cv_factor': dataset_config['cv_factor'],
            'seed': dataset_config['seed'],
            'privacy_rate': dataset_config['privacy_rate'],
        }
        jobs = generate_synthetic_jobs(**dataset_kwargs)
        get_timeslot(jobs)
        return jobs
    else:
        raise ValueError(
            f'Dataset {dataset_type} does not exist or has not been implemented yet.'
        )


def process_philly_jobs(philly_jobs: List['JobTrace']):
    """Converts entire Philly job trace into a list of simulator jobs.
    """
    jobs = philly_jobs.copy()
    # Remove invalid jobs (jobs that have not finished and jobs that failed/killed early)
    jobs = [j for j in jobs if j._run_time is not None and j.status == 'Pass']  # 对任务筛选，只保留了成功完成的任务，并按照提交时间排序
    jobs.sort(key=lambda j: j._submitted_time)

    # Arrival time for jobs
    start_time = jobs[0]._submitted_time  # 把整个系统开始的时间定义为第一个任务提交的时间（相对时间）
    arrival_times = [(j._submitted_time - start_time).total_seconds() / 3600.0  # .total_seconds()把时间转为秒，计算完后再转为以小时为单位
                     for j in jobs]

    # Run time for jobs
    run_times = [j._run_time / 60.0 for j in jobs]

    # Get GPU resources
    resources = []
    for j in jobs:
        gpu_count = sum(
            [len(node_dict['gpus']) for node_dict in j.attempts[-1]['detail']])  # 获取任务j最后一次（-1表示最后一次）尝试，得到gpu的个数
        resources.append({'GPUs': gpu_count})
    
    

    costs = [res['GPUs'] * run for res, run in zip(resources, run_times)]  # zip将资源列表和运行时间列表按顺序配对，每个索引上的内容就是对应索引任务的资源和时间

    return [Job(idx, arrival=arr, runtime=run, num_gpus=res['GPUs'], cost=cost, privacy=add_privacy()) \
            for idx, (arr, run, res, cost) in \
            enumerate(list(zip(arrival_times, run_times, resources, costs)))]
# 将arrival_times, run_times, resources, costs中的内容配对，并赋值给arr, run, res, cost

def generate_philly_gpu_jobs(philly_jobs: List['JobTrace'],
                             arrival_rate=32.0,
                             cv_factor=1.0,
                             total_jobs=300000,
                             seed=2024,
                             privacy_rate=0.2):
    """Generates Philly jobs based on a Poisson arrival distribution.

    Interarrival times follow an exponential distribution of 1/arrival_rate.
    Jobs are randomly sampled from the Philly job trace.
    """
    # 基于Philly生成模拟的GPU任务列表，使用泊松分布和其他随机采样方法生成任务到达时间、运行时间、资源需求等信息
    total_jobs = int(total_jobs)
    jobs = philly_jobs.copy()
    # Remove invalid jobs (jobs that have not finished and jobs that failed/killed early)
    jobs = [j for j in jobs if j._run_time is not None and j.status == 'Pass']
    jobs.sort(key=lambda j: j._submitted_time)

    # Arrival time for jobs
    np.random.seed(seed)  # 设置随机种子
    alpha = (1.0 / cv_factor)**2  # cv_factor影响任务突发性，arrival_rate控制任务到达的平均速率
    interarrival_times = np.array([
        np.random.gamma(shape=alpha, scale=1 / (alpha * arrival_rate))
        for _ in range(total_jobs - 1)
    ])  # 使用伽马分布生成到达时间间隔，一共total_jobs - 1个间隔
    # interarrival_times = np.random.exponential(scale=1 / arrival_rate,
    #                                            size=total_jobs - 1)
    interarrival_times = np.insert(interarrival_times, 0, 0)  # 在生成时间间隔的数组开头插入0，表达第一个任务到达时间为0。
    arrival_times = np.cumsum(interarrival_times)  # 从第一个任务开始，逐个增加每个的间隔时间，得到任务绝对到达时间
    # 例如，如果interarrival_times = [0, 1.5, 0.8, 2.1]，则arrival_times = [0, 1.5, 2.3, 4.4]

    # Run time for jobs
    run_times = []
    for j in jobs:
        run_time_hr = j._run_time / 60.0
        run_times.append(run_time_hr)

    # Get GPU resources
    resources = []
    for j in jobs:
        detail_dict = j.attempts[-1]['detail']
        gpu_count = sum([len(node_dict['gpus']) for node_dict in detail_dict])
        resources.append({'GPUs': gpu_count})
    np.random.seed(seed)
    job_indexes = np.random.choice(list(range(len(run_times))),
                                   size=total_jobs,
                                   #replace=True
                                   replace=False)  # 从原任务列表中随机采样total_jobs个任务，任务索引范围为len(run_times)
    proc_jobs = []
    for idx in range(total_jobs):
        job_idx = job_indexes[idx]
        resources_dict = resources[job_idx]
        runtime = run_times[job_idx]
        cost = resources_dict['GPUs'] * runtime
        proc_jobs.append(
            Job(idx,
                arrival=arrival_times[idx],
                runtime=runtime,
                num_gpus=resources_dict['GPUs'],
                cost=cost,
                privacy=add_privacy(privacy_rate=privacy_rate)))  # 创建任务对象
    return proc_jobs


def generate_helios_jobs(helios_jobs: List['HeliosJobTrace'],
                         arrival_rate=32.0,
                         cv_factor=1.0,
                         total_jobs=300000,
                         seed=2024,
                         privacy_rate=0.2):
    """Converts entire Helios job trace into a list of simulator jobs.
    """
    jobs = helios_jobs.copy()
    # Remove invalid jobs (jobs that have not finished and jobs that failed/killed early)
    jobs = [
        j for j in jobs
        if j._run_time is not None and (j.status in ['COMPLETED', 'TIMEOUT']) and (j.num_gpus > 0)
    ]
    jobs.sort(key=lambda j: j._submitted_time)

    # Arrival time for jobs
    np.random.seed(seed)
    alpha = (1.0 / cv_factor)**2
    interarrival_times = np.array([
        np.random.gamma(shape=alpha, scale=1 / (alpha * arrival_rate))
        for _ in range(total_jobs - 1)  # 
    ])
    # interarrival_times = np.random.exponential(scale=1 / arrival_rate,
    #                                            size=total_jobs - 1)
    interarrival_times = np.insert(interarrival_times, 0, 0)
    arrival_times = np.cumsum(interarrival_times)

    # Run time for jobs
    run_times = [j._run_time for j in jobs]

    # Get GPU resources
    resources = []
    for j in jobs:
        resources.append({'GPUs': j.num_gpus})

    costs = [res['GPUs']  * run
             for res, run in zip(resources, run_times)]

    np.random.seed(seed)
    job_indexes = np.random.choice(list(range(len(run_times))),
                                   size=total_jobs,
                                   #replace=True
                                   replace=False)
    proc_jobs = []
    for idx in range(total_jobs):
        job_idx = job_indexes[idx]
        resources_dict = resources[job_idx]
        runtime = run_times[job_idx]
        cost = costs[job_idx]
        proc_jobs.append(
            Job(idx,
                arrival=arrival_times[idx],
                runtime=runtime,
                num_gpus=resources_dict['GPUs'],
                cost=cost,
                privacy=add_privacy(privacy_rate=privacy_rate)))
    return proc_jobs


def generate_gpu_jobs(philly_jobs: List['JobTrace'],
                      arrival_rate=32.0,  # 参数表示每单位时间到达的平均任务数，即平均32个任务/小时
                      job_runtime=4.0,
                      total_jobs=200000,
                      seed=2024,
                      privacy_rate=0.2):
    """Generates GPU jobs based on a Poisson arrival distribution and exponential runtime distribution.
    """
    total_jobs = int(total_jobs)
    jobs = philly_jobs.copy()
    # Remove invalid jobs (jobs that have not finished and jobs that failed/killed early)
    jobs = [j for j in jobs if j._run_time is not None and j.status == 'Pass']
    jobs.sort(key=lambda j: j._submitted_time)

    # Arrival time for jobs
    np.random.seed(seed)
    interarrival_times = np.random.exponential(scale=1 / arrival_rate,
                                               size=total_jobs - 1)  # 设置时间间隔
    interarrival_times = np.insert(interarrival_times, 0, 0)
    arrival_times = np.cumsum(interarrival_times)  # 使用指数分布生成任务间隔时间，然后用累计和计算任务的到达时间，这样的实现是泊松过程的常见方法。

    # Run time for jobs
    run_times = np.random.exponential(scale=job_runtime, size=total_jobs)  # scale设置的是指数分布平均值，这里是4小时

    # Get GPU resources
    resources = []
    for j in jobs:
        detail_dict = j.attempts[-1]['detail']
        gpu_count = sum([len(node_dict['gpus']) for node_dict in detail_dict])
        resources.append({'GPUs': gpu_count})
    np.random.seed(seed)
    job_indexes = np.random.choice(list(range(len(resources))),
                                   size=total_jobs,
                                   replace=True)
    proc_jobs = []
    for idx in range(total_jobs):
        job_idx = job_indexes[idx]
        runtime = run_times[idx]
        resources_dict = resources[job_idx]
        cost = resources_dict['GPUs'] * runtime
        proc_jobs.append(
            Job(idx,
                arrival=arrival_times[idx],
                runtime=runtime,
                num_gpus=resources_dict['GPUs'],
                cost=cost,
                privacy=add_privacy(privacy_rate=privacy_rate)))
    return proc_jobs


def process_helios_jobs(helios_jobs: List['HeliosJobTrace']):
    """Converts entire Helios job trace into a list of simulator jobs.
    """
    jobs = helios_jobs.copy()
    # Remove invalid jobs (jobs that have not finished and jobs that failed/killed early)
    jobs = [
        j for j in jobs
        if j._run_time is not None and (j.status in ['COMPLETED', 'TIMEOUT']) and (j.num_gpus > 0)
    ]
    jobs.sort(key=lambda j: j._submitted_time)

    # Arrival time for jobs
    start_time = jobs[0]._submitted_time
    arrival_times = [(j._submitted_time - start_time).total_seconds() / 3600.0
                     for j in jobs]

    # Run time for jobs
    run_times = [j._run_time for j in jobs]

    # Get GPU resources
    resources = []
    for j in jobs:
        resources.append({'GPUs': j.num_gpus})

    costs = [(res['GPUs']) * run
             for res, run in zip(resources, run_times)]

    return [Job(idx, arrival=arr, runtime=run, num_gpus=res['GPUs'], cost=cost, privacy=add_privacy()) \
            for idx, (arr, run, res, cost) in \
            enumerate(list(zip(arrival_times, run_times, resources, costs)))]


def generate_synthetic_jobs(arrival_rate=8.0,
                            job_runtime=1.0,
                            cv_factor=1.0,  # 变异系数为1.0时，伽马分布退化为泊松分布
                            total_jobs=20000,
                            seed=2024,
                            privacy_rate = 0.2):  # 已添加任务的隐私敏感性
    """Generates GPU jobs based on a Poisson arrival distribution and exponential runtime distribution.
    """
    total_jobs = total_jobs

    # Arrival time for jobs
    np.random.seed(seed)
    alpha = (1.0 / cv_factor)**2
    interarrival_times = np.array([
        np.random.gamma(shape=alpha, scale=1 / (alpha * arrival_rate))
        for _ in range(total_jobs - 1)  # 泊松分布生成到达时间间隔（变异系数为1.0时伽马变泊松）
    ])
    # interarrival_times = np.random.exponential(scale=1 / arrival_rate,
    #                                            size=total_jobs - 1)
    interarrival_times = np.insert(interarrival_times, 0, 0)
    arrival_times = np.cumsum(interarrival_times)

    # Run time for jobs
    run_times = np.random.exponential(scale=job_runtime, size=total_jobs)  # 指数分布生成每个任务的运行时间

    # Get GPU resources
    proc_jobs = []
    categorical = [0.7, 0.15, 0.1, 0.05] # 这是需求各个GPU的任务的出现概率，如70%的任务使用1个GPU，15%使用2个GPU，依此类推
    sizes = [1, 2, 4, 8]  # 把GPU请求定义为2的幂次
    for idx in range(total_jobs):

        resources_dict = {'GPUs': np.random.choice(sizes, p=categorical)}
        temp = run_times[idx]
        cost = resources_dict['GPUs'] * temp
        #privacy = add_privacy(privacy_rate=privacy_rate)
        proc_jobs.append(
            Job(idx,
                arrival=arrival_times[idx],
                runtime=temp,
                num_gpus=resources_dict['GPUs'],
                cost=cost,
                privacy=add_privacy(privacy_rate=privacy_rate)))
        
    return proc_jobs

def add_privacy(privacy_rate = 0.2):
    # 默认的隐私率是0.2
    if np.random.rand() < privacy_rate:
        return True
    else:
        return False
    
def get_timeslot(jobs: List):
    for job in jobs:
        job.timeslot = int(job.arrival * 60) // 20


def main():
    # 调用函数来生成任务
    #helios_jobs = helios.load_helios_traces('~/HeliosData/data/Venus')
    #jobs = generate_helios_jobs(helios_jobs)
    dataset_config = {
        'dataset': 'philly_gen',
        'arrival_rate': 24,  # 设置的是每小时（3个时隙）到来的平均任务数量，因
        'cv_factor': 1.0,
        'total_jobs': 300,
        'seed': 2024,
        'privacy_rate': 0.2,
        'job_runtime': 4.0
        }
    jobs = load_processed_jobs(dataset_config)
    # 打印任务列表，检查每个任务的 privacy 属性
    for job in jobs[:100]:  # 只打印前100个
        print(job.__repr__)


# 确保只在脚本直接运行时调用main函数，而不是作为模块导入时
if __name__ == "__main__":
    main()

