import os
import random

from time import process_time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback

import matchings


class CourseRecEnv(gym.Env):
    # The CourseRecEnv class is a gym environment that simulates the recommendation of courses to learners. It is used to train the Reinforce model.
    def __init__(self, dataset, threshold=0.8, k=3):
        self.dataset = dataset
        self.nb_skills = len(dataset.skills)
        self.mastery_levels = [
            elem for elem in list(dataset.mastery_levels.values()) if elem > 0
        ]
        self.max_level = max(self.mastery_levels)
        self.nb_courses = len(dataset.courses)
        # get the minimum and maximum number of skills of the learners using np.nonzero
        self.min_skills = min(np.count_nonzero(self.dataset.learners, axis=1))
        self.max_skills = max(np.count_nonzero(self.dataset.learners, axis=1))
        self.threshold = threshold
        self.k = k
        # The observation space is a vector of length nb_skills that represents the learner's skills
        self.observation_space = gym.spaces.Box(
            low=0, high=self.max_level, shape=(self.nb_skills,), dtype=np.int32
        )
        # The action space is a discrete space of size nb_courses that represents the courses to be recommended
        self.action_space = gym.spaces.Discrete(self.nb_courses)

    def _get_obs(self):
        """Method required by the gym environment. It returns the current observation of the environment.

        Returns:
            np.array: the current observation of the environment, that is the learner's skills
        """
        return self._agent_skills

    def _get_info(self):
        """Method required by the gym environment. It returns the current info of the environment.

        Returns:
            dict: the current info of the environment, that is the number of applicable jobs
        """

        return {
            "nb_applicable_jobs": self.dataset.get_nb_applicable_jobs(
                self._agent_skills, threshold=self.threshold
            )
        }

    def get_random_learner(self):
        """Creates a random learner with a random number of skills and levels. This method is used to initialize the environment.

        Returns:
            np.array: the initial observation of the environment, that is the learner's initial skills
        """
        # Randomly choose the number of skills the agent has randomly
        n_skills = random.randint(self.min_skills, self.max_skills)

        # Initialize the skills array with zeros
        initial_skills = np.zeros(self.nb_skills, dtype=np.int32)

        # Choose unique skill indices without replacement
        skill_indices = np.random.choice(self.nb_skills, size=n_skills, replace=False)

        # Assign random mastery levels to these skills, levels can repeat
        initial_skills[skill_indices] = np.random.choice(
            self.mastery_levels, size=n_skills, replace=True
        )
        return initial_skills

    def reset(self, seed=None, learner=None):
        """Method required by the gym environment. It resets the environment to its initial state.

        Args:
            seed (int, optional): Random seed. Defaults to None.
            learner (list, optional): Learner to initialize the environment with, if None, the environment is initialized with a random learner. Defaults to None.

        Returns:
            _type_: _description_
        """
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        if learner is not None:
            self._agent_skills = learner
        else:
            self._agent_skills = self.get_random_learner()
        self.nb_recommendations = 0
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        """Method required by the gym environment. It performs the action in the environment and returns the new observation, the reward, whether the episode is terminated and additional information.

        Args:
            action (int): the course to be recommended

        Returns:
            tuple: the new observation, the reward, whether the episode is terminated, additional information
        """
        # Update the agent's skills with the course provided_skills

        course = self.dataset.courses[action]
        learner = self._agent_skills

        required_matching = matchings.learner_course_required_matching(learner, course)
        provided_matching = matchings.learner_course_provided_matching(learner, course)
        if required_matching < self.threshold or provided_matching >= 1.0:
            observation = self._get_obs()
            reward = -1
            terminated = True
            info = self._get_info()
            return observation, reward, terminated, False, info

        self._agent_skills = np.maximum(self._agent_skills, course[1])

        observation = self._get_obs()
        info = self._get_info()
        reward = info["nb_applicable_jobs"]
        self.nb_recommendations += 1
        terminated = self.nb_recommendations == self.k

        return observation, reward, terminated, False, info


class EvaluateCallback(BaseCallback):
    # The EvaluateCallback class is a callback that evaluates the model at regular intervals during the training.
    def __init__(self, eval_env, eval_freq, all_results_filename, verbose=1):
        super(EvaluateCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.all_results_filename = all_results_filename
        self.mode = "w"

    def _on_step(self):
        """Method required by the callback. It is called at each step of the training. It evaluates the model every eval_freq steps.

        Returns:
            bool: Always returns True to continue training
        """
        if self.n_calls % self.eval_freq == 0:
            time_start = process_time()
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
            time_end = process_time()
            print(
                f"Iteration {self.n_calls}. Average jobs: {avg_jobs / len(self.eval_env.dataset.learners)} Time: {time_end - time_start}"
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
