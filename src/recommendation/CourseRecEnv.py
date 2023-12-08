import random

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback

import matchings


class CourseRecEnv(gym.Env):
    def __init__(self, dataset, min_skills=0, max_skills=10, threshold=0.8, k=3):
        self.dataset = dataset
        self.nb_skills = len(dataset.skills) // 2
        self.max_level = max(dataset.mastery_levels)
        self.nb_courses = len(dataset.courses)
        self.min_skills = min_skills
        self.max_skills = max_skills
        self.threshold = threshold
        self.k = k
        self.observation_space = gym.spaces.Box(
            low=0, high=self.max_level, shape=(self.nb_skills,), dtype=np.int32
        )

        self.action_space = gym.spaces.Discrete(self.nb_courses)

    def _get_obs(self):
        return self._agent_skills

    def _get_info(self):
        learner = dict()
        learner["possessed_skills"] = dict()
        for skill_id, level in enumerate(self._agent_skills):
            if level > 0:
                skill_name = self.dataset.skills[str(skill_id)]
                learner["possessed_skills"][skill_name] = level

        return {
            "nb_applicable_jobs": self.dataset.get_nb_applicable_jobs(
                learner, threshold=self.threshold
            )
        }

    def obs_to_learner(self):
        learner = dict()
        learner["possessed_skills"] = dict()
        for skill_id, level in enumerate(self._agent_skills):
            if level > 0:
                skill_name = self.dataset.skills[str(skill_id)]
                learner["possessed_skills"][skill_name] = level
        return learner

    def learner_to_obs(self, learner):
        obs = np.zeros(self.nb_skills, dtype=np.int32)
        for skill, level in learner["possessed_skills"].items():
            skill_id = int(self.dataset.skills[skill])
            obs[skill_id] = level
        return obs

    def get_random_learner(self):
        # Choose the number of skills the agent has randomly
        n_skills = random.randint(self.min_skills, self.max_skills)
        initial_skills = np.zeros(self.nb_skills, dtype=np.int32)
        skills = np.random.choice(self.nb_skills, size=n_skills, replace=False)
        levels = np.random.choice(
            self.dataset.mastery_levels,
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
        self.current_attractiveness = info["nb_applicable_jobs"]
        return observation, info

    def step(self, action):
        # Update the agent's skills with the course provided_skills

        course = self.dataset.courses[str(action)]
        learner = self.obs_to_learner()

        required_matching = matchings.learner_course_required_matching(learner, course)
        provided_matching = matchings.learner_course_provided_matching(learner, course)
        if required_matching < self.threshold or provided_matching >= 1.0:
            observation = self._get_obs()
            reward = -1
            terminated = True
            info = self._get_info()
            return observation, reward, terminated, False, info

        for skill, level in course["provided_skills"].items():
            skill_id = int(self.dataset.skills[skill])
            if level > self._agent_skills[skill_id]:
                self._agent_skills[skill_id] = level

        observation = self._get_obs()
        info = self._get_info()
        reward = info["nb_applicable_jobs"] - self.current_attractiveness
        self.current_attractiveness = info["nb_applicable_jobs"]
        self.nb_recommendations += 1
        terminated = self.nb_recommendations == self.k

        return observation, reward, terminated, False, info


class EvaluateCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq, verbose=1):
        super(EvaluateCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            avg_jobs = 0
            for learner in self.eval_env.dataset.learners.values():
                self.eval_env.reset(learner=learner)
                done = False
                while not done:
                    obs = self.eval_env._get_obs()
                    old_attractiveness = self.eval_env._get_info()["nb_applicable_jobs"]
                    action, _state = self.model.predict(obs, deterministic=True)
                    obs, reward, done, _, info = self.eval_env.step(action)
                avg_jobs += info["nb_applicable_jobs"]
            print(self.n_calls, avg_jobs / len(self.eval_env.dataset.learners))
        return True  # Return True to continue training
