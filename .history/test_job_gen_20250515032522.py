import json
from job_generator import load_processed_jobs

dataset_config_ex = {  
    'dataset': 'synthetic_ex',
    'arrival_rate': 90,
    'cv_factor': 1.0,
    'total_jobs': 1000,
    'seed': 2024,
    'privacy_rate': 0.2,
    'job_runtime': 1.0
}
init_jobs_ex = load_processed_jobs(dataset_config_ex)

dataset_config = {  
    'dataset': 'synthetic',
    'arrival_rate': 90,
    'cv_factor': 1.0,
    'total_jobs': 1000,
    'seed': 2024,
    'privacy_rate': 0.2,
    'job_runtime': 1.0
}
init_jobs = load_processed_jobs(dataset_config)

# 写入到文件
output_data = {
    "init_jobs_ex": init_jobs_ex,
    "init_jobs": init_jobs
}

with open("output_jobs.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)
