# ./greedy.sh &
# ./optimal.sh &
# ./dqn.sh &
# ./ppo.sh &
# ./a2c.sh &


for k in 4 5
do
    total_steps=$((1000000 * k))
    time python src/recommendation/pipeline.py --config config/dataset_V2.yaml --threshold 0.8 --model dqn -k $k --total_steps $total_steps --eval_freq 50000 --nb_runs 1
    # time python src/recommendation/pipeline.py --config config/dataset_V2.yaml --threshold 0.8 --model a2c -k $k --total_steps $total_steps --eval_freq 50000 --nb_runs 1
    # time python src/recommendation/pipeline.py --config config/dataset_V2.yaml --threshold 0.8 --model ppo -k $k --total_steps $total_steps --eval_freq 50000 --nb_runs 1
done
