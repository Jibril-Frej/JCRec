import argparse

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
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Threshold value for recommendation",
    )
    parser.add_argument("-k", type=int, default=1, help="Number of recommendations")
    parser.add_argument(
        "--model",
        default="greedy",
        choices=["greedy", "optimal", "dqn", "a2c", "ppo"],
        help="Model to use for recommendation",
    )
    parser.add_argument(
        "--total_steps",
        type=int,
        default=1000,
        help="Total steps for reinforcement learning models",
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=100,
        help="Evaluation frequency for reinforcement learning models",
    )
    parser.add_argument(
        "--nb_runs", type=int, default=1, help="Number of runs to perform"
    )

    args = parser.parse_args()

    print(args)

    model_classes = {
        "greedy": Greedy,
        "optimal": Optimal,
        "reinforce": Reinforce,  # assuming other models also use the Reinforce class
    }

    for run in range(args.nb_runs):
        dataset = create_and_print_dataset(args.config)
        if args.model in ["greedy", "optimal"]:
            recommender = model_classes[args.model](dataset, args.threshold)
            recommendation_method = getattr(recommender, f"{args.model}_recommendation")
            recommendation_method(args.k, run)
        else:
            recommender = Reinforce(
                dataset,
                args.model,
                args.k,
                args.threshold,
                run,
                args.total_steps,
                args.eval_freq,
            )
            recommender.reinforce_recommendation()


if __name__ == "__main__":
    main()
