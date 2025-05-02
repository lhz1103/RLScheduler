class Job(object):
    def __init__(self,
                 idx: int,
                 arrival: float = 0.0,
                 runtime: float = 0.0,
                 deadline: float = 0.0,
                 num_gpus: int = 0,
                 cost: float = 0.0,
                 privacy: bool = False):
        self.idx = idx
        # Original arrival time for job.
        self.arrival = arrival  # 任务的到达时间是相对的，以第一个任务到达时间为0，后续任务和它的差值即到达时间，单位是小时
        self.runtime = runtime  # 单位也是小时

        self.deadline = deadline  # 等待时间上限
        self.waiting = False  # 记录是否已经在等待
        self.waitingtime = 0.0 # 记录已经等待的时间

        self.num_gpus = num_gpus

        self.cost = cost

        # State of the Job
        self.state = None
        # Starting time of the job on the local cluster, if none, the job was ran on the cloud.
        self.start = None

        # Keeps track of which GPU(s) the job ran on.键代表在第几个节点，值是一个列表，表示任务占用该节点的GPU索引
        self.allocated_gpus = {}

        # 一个列表，也是记录任务占用节点的资源情况，索引代表节点id，值代表占用GPU的个数，这是我自定义的，感觉更简洁
        self.assigned_gpus = []

        # 根据分配的服务器数量调整实际的运行时间，也是我自定义
        self.actual_time = 0.0

        # 任务所属的时隙
        self.timeslot = -1

        # 任务完成的时隙
        self.finished_timeslot = -1

        # For backfill scheduling, job immediately executed after Job idx `block_job_idx` completes.
        self.block_job_idx = None

        # This field keeps track of the total starved space a job has incurred due to preemption.
        # This is to prevent chain preemptions.
        self.starved_space = 0


        # This field keeps track if the job has been on the cloud before and was preemepted from running on cloud.
        self.preempt_cloud = False
        # New arrival time for job.
        self.new_arrival = -1

        # 标记任务是否是隐私敏感的
        self.privacy = privacy

    def __eq__(self, other):
        return self.idx == other.idx

    def __hash__(self):
        return hash(str(self.idx))

    def set_deadline(self, deadline):
        self.deadline = deadline

    def __repr__(self):
        return f'Job(idx={self.idx}, state={self.state}, gpus={self.num_gpus}, arr={self.arrival}, timeslot={self.timeslot}, run={self.runtime}, actual={self.actual_time}, deadline={self.deadline}, start={self.start}, privacy={self.privacy}, assigned={self.assigned_gpus})\n'
