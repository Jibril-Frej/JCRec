import argparse

from Dataset import Dataset
from Greedy import Greedy
from Optimal import Optimal


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path")
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("-k", type=int, default=1)
    parser.add_argument(
        "--model", default="greedy", choices=["greedy", "optimal", "reinforcement"]
    )

    args = parser.parse_args()

    dataset_path = args.dataset_path
    threshold = args.threshold
    model = args.model
    k = args.k

    dataset = Dataset(dataset_path)

    if model == "greedy":
        greedy_recommender = Greedy(dataset, threshold)
        greedy_recommender.greedy_recommendation(k)

    if model == "optimal":
        optimal_recommender = Optimal(dataset, threshold)
        optimal_recommender.optimal_recommendation(k)


if __name__ == "__main__":
    main()
