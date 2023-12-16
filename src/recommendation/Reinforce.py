import os
import json

from copy import deepcopy
from tqdm import tqdm
from stable_baselines3 import DQN, A2C, PPO

from Dataset import Dataset
from CourseRecEnv import CourseRecEnv, EvaluateCallback


class Reinforce:
    def __init__(self, dataset, model, k, threshold, total_steps=1000, eval_freq=100):
        self.dataset = dataset
        self.model_name = model
        self.k = k
        self.threshold = threshold
        self.total_steps = total_steps
        self.eval_freq = eval_freq
        self.train_env = CourseRecEnv(dataset, threshold=self.threshold, k=self.k)
        self.eval_env = CourseRecEnv(dataset, threshold=self.threshold, k=self.k)
        self.get_model()
        self.eval_callback = EvaluateCallback(self.eval_env, eval_freq=self.eval_freq)

    def get_model(self):
        if self.model_name == "dqn":
            self.model = DQN(env=self.train_env, verbose=0, policy="MlpPolicy")
        elif self.model_name == "a2c":
            self.model = A2C(
                env=self.train_env, verbose=0, policy="MlpPolicy", device="cpu"
            )
        elif self.model_name == "ppo":
            self.model = PPO(env=self.train_env, verbose=0, policy="MlpPolicy")

    def update_learner_profile(self, learner, course):
        for cskill, clevel in course[1]:
            found = False
            i = 0
            while not found and i < len(learner):
                lskill, llevel = learner[i]
                if cskill == lskill:
                    learner[i] = (lskill, max(llevel, clevel))
                    found = True
                i += 1
            if not found:
                learner.append((cskill, clevel))

    def reinforce_recommendation(self):
        results = dict()

        avg_l_attrac = self.dataset.get_avg_learner_attractiveness()
        print(f"The average attractiveness of the learners is {avg_l_attrac:.2f}")

        results["original_attractiveness"] = avg_l_attrac

        avg_app_j = self.dataset.get_avg_applicable_jobs(self.threshold)
        print(f"The average nb of applicable jobs per learner is {avg_app_j:.2f}")

        results["original_applicable_jobs"] = avg_app_j

        self.model.learn(total_timesteps=self.total_steps, callback=self.eval_callback)

        recommendations = dict()
        for i, learner in enumerate(tqdm(self.dataset.learners)):
            self.eval_env.reset(learner=learner)
            done = False
            recommendations[i] = []
            while not done:
                obs = self.eval_env._get_obs()
                action, _state = self.model.predict(obs, deterministic=True)
                obs, reward, done, _, info = self.eval_env.step(action)
                if reward != -1:
                    recommendations[i].append(action.item())
            for course in recommendations[i]:
                self.update_learner_profile(learner, self.dataset.courses[course])
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
                    self.model_name + "_" + str(self.k) + ".json",
                ),
                "w",
            ),
            indent=4,
        )
