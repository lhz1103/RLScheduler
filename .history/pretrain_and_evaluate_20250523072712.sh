#!/usr/bin/env bash
# 文件名：run_two_steps.sh
# 用途：先训练，再测试；都使用 nohup，并把输出分别写入日志文件

# 1. 后台启动训练脚本
nohup python evaluate_pretrain.py > RL_pretrain_log.txt 2>&1 &
train_pid=$!              # 保存后台进程的 PID

# 2. 等待训练脚本结束
wait "$train_pid"

# 3. 训练结束后启动测试脚本
nohup python HeuristicScheduling_new_order_test.py > Heur_new_order_8nodes_log.txt 2>&1 &
