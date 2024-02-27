import os
import argparse

import yaml

from Dataset import Dataset
from Greedy import Greedy
from Optimal import Optimal
from Reinforce import Reinforce


def create_and_print_dataset(config):
    """Create and print the dataset."""
    dataset = Dataset(config)
    print(dataset)
    return dataset


def main():
    """Run the recommender system based on the provided model and parameters."""
    parser = argparse.ArgumentParser(description="Run recommender models.")

    parser.add_argument("--config", help="Path to the configuration file")

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model_classes = {
        "greedy": Greedy,
        "optimal": Optimal,
        "reinforce": Reinforce,
    }

    for run in range(config["nb_runs"]):
        dataset = create_and_print_dataset(config)
        # If the model is greedy or optimal, we use the corresponding class defined in Greedy.py and Optimal.py
        if config["model"] in ["greedy", "optimal"]:
            recommender = model_classes[config["model"]](dataset, config["threshold"])
            recommendation_method = getattr(
                recommender, f'{config["model"]}_recommendation'
            )
            recommendation_method(config["k"], run)
        # Otherwise, we use the Reinforce class, described in Reinforce.py
        else:
            recommender = Reinforce(
                dataset,
                config["model"],
                config["k"],
                config["threshold"],
                run,
                config["total_steps"],
                config["eval_freq"],
            )
            recommender.reinforce_recommendation()


if __name__ == "__main__":
    main()
