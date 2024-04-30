import os
import json

import numpy as np
from time import process_time
from stable_baselines3 import DQN, A2C, PPO

from CourseRecEnv import CourseRecEnv, EvaluateCallback


class Reinforce:
    def __init__(
        self, dataset, model, k, threshold, run, total_steps=1000, eval_freq=100
    ):
        self.dataset = dataset
        self.model_name = model
        self.k = k
        self.threshold = threshold
        self.run = run
        self.total_steps = total_steps
        self.eval_freq = eval_freq
        # Create the training and evaluation environments
        self.train_env = CourseRecEnv(dataset, threshold=self.threshold, k=self.k)
        self.eval_env = CourseRecEnv(dataset, threshold=self.threshold, k=self.k)
        self.get_model()
        self.all_results_filename = (
            "all_"
            + self.model_name
            + "_nbskills_"
            + str(len(self.dataset.skills))
            + "_k_"
            + str(self.k)
            + "_run_"
            + str(run)
            + ".txt"
        )
        self.final_results_filename = (
            "final_"
            + self.model_name
            + "_nbskills_"
            + str(len(self.dataset.skills))
            + "_k_"
            + str(self.k)
            + "_run_"
            + str(self.run)
            + ".json"
        )

        self.eval_callback = EvaluateCallback(
            self.eval_env,
            eval_freq=self.eval_freq,
            all_results_filename=self.all_results_filename,
        )

    def get_model(self):
        """Sets the model to be used for the recommendation. The model is from stable-baselines3 and is chosen based on the model_name attribute."""
        if self.model_name == "dqn":
            self.model = DQN(env=self.train_env, verbose=0, policy="MlpPolicy")
        elif self.model_name == "a2c":
            self.model = A2C(
                env=self.train_env, verbose=0, policy="MlpPolicy", device="cpu"
            )
        elif self.model_name == "ppo":
            self.model = PPO(env=self.train_env, verbose=0, policy="MlpPolicy")

    def update_learner_profile(self, learner, course):
        """Updates the learner's profile with the skills and levels of the course.

        Args:
            learner (list): list of skills and mastery level of the learner
            course (list): list of required (resp. provided) skills and mastery level of the course
        """
        learner = np.maximum(learner, course[1])
        return learner

    def reinforce_recommendation(self):
        """Train and evaluates the reinforcement learning model to make recommendations for every learner in the dataset. The results are saved in a json file."""
        results = dict()

        avg_l_attrac = self.dataset.get_avg_learner_attractiveness()
        print(f"The average attractiveness of the learners is {avg_l_attrac:.2f}")

        results["original_attractiveness"] = avg_l_attrac

        avg_app_j = self.dataset.get_avg_applicable_jobs(self.threshold)
        print(f"The average nb of applicable jobs per learner is {avg_app_j:.2f}")

        results["original_applicable_jobs"] = avg_app_j

        # Train the model
        self.model.learn(total_timesteps=self.total_steps, callback=self.eval_callback)

        # Evaluate the model
        time_start = process_time()
        recommendations = dict()
        for i, learner in enumerate(self.dataset.learners):
            self.eval_env.reset(learner=learner)
            done = False
            index = self.dataset.learners_index[i]
            recommendation_sequence = []
            while not done:
                obs = self.eval_env._get_obs()
                action, _state = self.model.predict(obs, deterministic=True)
                obs, reward, done, _, info = self.eval_env.step(action)
                if reward != -1:
                    recommendation_sequence.append(action.item())
            for course in recommendation_sequence:
                self.dataset.learners[i] = self.update_learner_profile(
                    learner, self.dataset.courses[course]
                )

            recommendations[index] = [
                self.dataset.courses_index[course_id]
                for course_id in recommendation_sequence
            ]

        time_end = process_time()
        avg_recommendation_time = (time_end - time_start) / len(self.dataset.learners)

        print(f"Average Recommendation Time: {avg_recommendation_time:.4f} seconds")

        results["avg_recommendation_time"] = avg_recommendation_time
        avg_l_attrac = self.dataset.get_avg_learner_attractiveness()

        print(f"The new average attractiveness of the learners is {avg_l_attrac:.2f}")

        results["new_attractiveness"] = avg_l_attrac

        avg_app_j = self.dataset.get_avg_applicable_jobs(self.threshold)
        print(f"The new average nb of applicable jobs per learner is {avg_app_j:.2f}")

        results["new_applicable_jobs"] = avg_app_j

        results["recommendations"] = recommendations

        json.dump(
            results,
            open(
                os.path.join(
                    self.dataset.config["results_path"],
                    self.final_results_filename,
                ),
                "w",
            ),
            indent=4,
        )
