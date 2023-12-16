import argparse

from Dataset import Dataset
from Greedy import Greedy
from Optimal import Optimal
from Reinforce import Reinforce


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config")
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("-k", type=int, default=1)
    parser.add_argument(
        "--model", default="greedy", choices=["greedy", "optimal", "dqn", "a2c", "ppo"]
    )
    parser.add_argument("--total_steps", type=int, default=1000)
    parser.add_argument("--eval_freq", type=int, default=100)

    args = parser.parse_args()

    config = args.config

    dataset = Dataset(args.config)
    print(dataset)

    if args.model == "greedy":
        greedy_recommender = Greedy(dataset, args.threshold)
        greedy_recommender.greedy_recommendation(args.k)

    elif args.model == "optimal":
        optimal_recommender = Optimal(dataset, args.threshold)
        optimal_recommender.optimal_recommendation(args.k)

    else:
        reinforce_recommender = Reinforce(
            dataset,
            args.model,
            args.k,
            args.threshold,
            args.total_steps,
            args.eval_freq,
        )
        reinforce_recommender.reinforce_recommendation()


if __name__ == "__main__":
    main()
