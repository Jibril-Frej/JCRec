import argparse

from Dataset import Dataset
from Greedy import Greedy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path")
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("-k", type=int, default=1)

    args = parser.parse_args()

    dataset_path = args.dataset_path
    threshold = args.threshold
    k = args.k

    dataset = Dataset(dataset_path)

    greedy_recommender = Greedy(dataset, threshold)
    greedy_recommender.greedy_recommendation(k)


if __name__ == "__main__":
    main()
