import os
import random

import time as time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback

import matchings


class CourseRecEnv(gym.Env):
    def __init__(self, dataset, threshold=0.8, k=3):
        self.dataset = dataset
        self.nb_skills = len(dataset.skills)
        self.mastery_levels = [
            elem for elem in list(dataset.mastery_levels.values()) if elem > 0
        ]
        self.max_level = max(self.mastery_levels)
        self.nb_courses = len(dataset.courses)
        self.min_skills = min([len(learner) for learner in dataset.learners])
        self.max_skills = max([len(learner) for learner in dataset.learners])
        self.threshold = threshold
        self.k = k
        self.observation_space = gym.spaces.Box(
            low=0, high=self.max_level, shape=(self.nb_skills,), dtype=np.int32
        )

        self.action_space = gym.spaces.Discrete(self.nb_courses)

    def _get_obs(self):
        return self._agent_skills

    def _get_info(self):
        learner = self.obs_to_learner()

        return {
            "nb_applicable_jobs": self.dataset.get_nb_applicable_jobs(
                learner, threshold=self.threshold
            )
        }

    def obs_to_learner(self):
        learner = []
        for skill, level in enumerate(self._agent_skills):
            if level > 0:
                learner.append((skill, level))
        return learner

    def learner_to_obs(self, learner):
        obs = np.zeros(self.nb_skills, dtype=np.int32)
        for skill, level in learner:
            obs[skill] = level
        return obs

    def get_random_learner(self):
        # Choose the number of skills the agent has randomly
        n_skills = random.randint(self.min_skills, self.max_skills)
        initial_skills = np.zeros(self.nb_skills, dtype=np.int32)
        skills = np.random.choice(self.nb_skills, size=n_skills, replace=False)
        levels = np.random.choice(
            self.mastery_levels,
            n_skills,
            replace=True,
        )
        for skill, level in zip(skills, levels):
            initial_skills[skill] = level
        return initial_skills

    def reset(self, seed=None, options=None, learner=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        if learner is not None:
            self._agent_skills = self.learner_to_obs(learner)
        else:
            self._agent_skills = self.get_random_learner()
        self.nb_recommendations = 0
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        # Update the agent's skills with the course provided_skills

        course = self.dataset.courses[action]
        learner = self.obs_to_learner()

        required_matching = matchings.learner_course_required_matching(learner, course)
        provided_matching = matchings.learner_course_provided_matching(learner, course)
        if required_matching < self.threshold or provided_matching >= 1.0:
            observation = self._get_obs()
            reward = -1
            terminated = True
            info = self._get_info()
            return observation, reward, terminated, False, info

        for skill, level in course[1]:
            self._agent_skills[skill] = max(self._agent_skills[skill], level)

        observation = self._get_obs()
        info = self._get_info()
        reward = info["nb_applicable_jobs"]
        self.nb_recommendations += 1
        terminated = self.nb_recommendations == self.k

        return observation, reward, terminated, False, info


class EvaluateCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq, all_results_filename, verbose=1):
        super(EvaluateCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.all_results_filename = all_results_filename
        self.mode = "w"

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            time_start = time.time()
            avg_jobs = 0
            for learner in self.eval_env.dataset.learners:
                self.eval_env.reset(learner=learner)
                done = False
                tmp_avg_jobs = self.eval_env._get_info()["nb_applicable_jobs"]
                while not done:
                    obs = self.eval_env._get_obs()
                    action, _state = self.model.predict(obs, deterministic=True)
                    obs, reward, done, _, info = self.eval_env.step(action)
                    if reward != -1:
                        tmp_avg_jobs = reward
                avg_jobs += tmp_avg_jobs
            time_end = time.time()
            print(
                self.n_calls,
                avg_jobs / len(self.eval_env.dataset.learners),
                time_end - time_start,
            )
            with open(
                os.path.join(
                    self.eval_env.dataset.config["results_path"],
                    self.all_results_filename,
                ),
                self.mode,
            ) as f:
                f.write(
                    str(self.n_calls)
                    + " "
                    + str(avg_jobs / len(self.eval_env.dataset.learners))
                    + " "
                    + str(time_end - time_start)
                    + "\n"
                )
            if self.mode == "w":
                self.mode = "a"
        return True  # Return True to continue training
