for k in 1 2 3 4 5
do
    python src/recommendation/pipeline.py --config config/dataset_V2.yaml --threshold 0.8 --model ppo -k $k --total_steps 1000000 --eval_freq 100000 --nb_runs 3> data/V2/results/stdout_ppo_$k &

done