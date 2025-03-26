#!/bin/bash

# 定义要检查的端口数组
PORTS=(5010 5173 9200 11434)

for PORT in "${PORTS[@]}"; do
    # 使用lsof查找占用端口的进程PID
    PID=$(lsof -ti :$PORT)

    if [[ -n "$PID" ]]; then
        echo "端口 $PORT 被进程 $PID 占用，正在终止..."
        kill -9 $PID
        if [[ $? -eq 0 ]]; then
            echo "成功终止进程 $PID"
        else
            echo "终止进程 $PID 失败"
        fi
    else
        echo "端口 $PORT 未被占用"
    fi
done