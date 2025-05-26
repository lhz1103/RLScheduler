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


# 写入文本文件
with open("jobs_output.txt", "w", encoding="utf-8") as f:
    f.write("=== init_jobs_ex ===\n")
    for job in init_jobs_ex:
        f.write(repr(job))  # 使用 __repr__ 格式化输出
        f.write("\n")
