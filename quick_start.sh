#!/bin/bash

es/bin/elasticsearch & pid1=$!
ollama serve & pid2=$!
python py/main.py & pid3=$!
npm run dev & pid4=$!

trap 'pkill $pid1 $pid2 $pid3 $pid4; exit' SIGINT

wait $pid1 || echo "elasticsearch failed"
wait $pid2 || echo "ollama serve failed"
wait $pid3 || echo "python main.py failed"
wait $pid4 || echo "npm run dev failed"
