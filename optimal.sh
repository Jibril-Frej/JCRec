for k in 1 2
do
    echo "k = $k"
    python src/recommendation/pipeline.py --config config/dataset_V2.yaml --threshold 0.8 --model optimal -k $k --total_steps 1000000 --eval_freq 100000 --nb_runs 1

done
```