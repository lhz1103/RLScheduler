import csv
from dateutil import parser
from multiprocessing import Pool
import os


class HeliosJobTrace:  # 封装任务信息
    """Encapsulates a job."""
    def __init__(self, idx, num_gpus, submitted_time,
                 run_time, status):
        self._idx = idx
        self._num_gpus = num_gpus
        self._submitted_time = submitted_time
        self._run_time = run_time
        self._status = status

    def __repr__(self):
        return (f'Job(idx={self._idx},\n'
                f'status={self._status},\n'
                f'num_gpus = {self._num_gpus},\n'
                f'submitted_time = {self._submitted_time},\n'
                f'run_time={self._run_time}\n')

    @property
    def idx(self):
        return self._idx

    @property
    def num_gpus(self):
        return self._num_gpus

    @property
    def submitted_time(self):
        return self._submitted_time

    @property
    def run_time(self):
        return self._run_time

    @property
    def status(self):
        return self._status


def load_helios_traces(trace_dir: str):
    trace_dir = os.path.abspath(os.path.expanduser(trace_dir))
    cluster_job_log_path = os.path.join(trace_dir, 'cluster_log.csv')
    jobs = []
    counter = 0
    with open(cluster_job_log_path, 'r') as f:
        csv_reader = csv.DictReader(f)  # csv文件转为字典
        # Iterate over each row in the CSV file
        for row in csv_reader:
            # Do something with the data in each row
            jobs.append(
                HeliosJobTrace(idx=counter,
                               num_gpus=int(row['gpu_num']),
                               submitted_time=parser.parse(row['submit_time']),
                               run_time=float(row['duration']) / 3600.0,
                               status=row['state']))
        counter += 1
    jobs = sorted(jobs, key=lambda t: t._submitted_time)
    return jobs
