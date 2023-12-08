import os
import json

import yaml
from collections import Counter
import pandas as pd

import matchings


class Dataset:
    def __init__(self, config):
        self.config = config
        self.skills = None
        self.mastery_levels = None
        self.years = None
        self.learners = None
        self.jobs = None
        self.jobs_inverted_index = None
        self.courses = None
        self.skill_supply = None
        self.skill_demand = None
        self.skills_attractiveness = None
        self.learners_attractiveness = None
        self.load_data()
        # self.get_all_market_metrics()

        # self.get_jobs_inverted_index()

    def load_data(self):
        with open(self.config, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        self.skills = pd.read_csv(self.config["taxonomy_path"]).unique_id.to_list()

        # data = [
        #     json.load(open(os.path.join(self.dataset_path, fname)))
        #     for fname in filenames
        # ]

        # (
        #     self.skills,
        #     self.mastery_levels,
        #     self.years,
        #     self.learners,
        #     self.jobs,
        #     self.courses,
        # ) = data

    def get_jobs_inverted_index(self):
        self.jobs_inverted_index = dict()
        for id_j, job in self.jobs.items():
            for skill in job["required_skills"]:
                if skill not in self.jobs_inverted_index:
                    self.jobs_inverted_index[skill] = set()
                self.jobs_inverted_index[skill].add(id_j)

    def get_all_market_metrics(self):
        self.get_skill_supply()
        self.get_skill_demand()
        self.get_all_skills_attractiveness()
        self.get_all_learners_attractiveness()

    def get_skill_supply(self):
        self.skill_supply = {year: Counter() for year in self.years}
        for learner in self.learners.values():
            for skill, mastery_level in learner["possessed_skills"].items():
                for sublevel in range(mastery_level, 0, -1):
                    self.skill_supply[learner["year"]][(skill, sublevel)] += 1
        for year, supply in self.skill_supply.items():
            for skill, mastery_level in self.skill_supply[year]:
                self.skill_supply[year][(skill, mastery_level)] /= supply.total()

    def get_skill_demand(self):
        self.skill_demand = {year: Counter() for year in self.years}
        max_level = max(self.mastery_levels)
        for job in self.jobs.values():
            for skill, mastery_level in job["required_skills"].items():
                for sublevel in range(mastery_level, max_level + 1):
                    self.skill_demand[job["year"]][(skill, sublevel)] += 1

        for year, demand in self.skill_demand.items():
            for skill, level in self.skill_demand[year]:
                self.skill_demand[year][(skill, level)] /= demand.total()

    def get_skill_attractiveness(self, skill, mastery_level):
        skill_attractiveness = 0
        normalization_factor = 0
        for i, year in enumerate(self.years):
            skill_attractiveness += (
                self.skill_demand[year][(skill, mastery_level)]
                * (1 - self.skill_supply[year][(skill, mastery_level)])
                / (i + 1)
            )
            normalization_factor += 1 / (i + 1)
        return skill_attractiveness / normalization_factor

    def get_all_skills_attractiveness(self):
        self.skills_attractiveness = Counter()
        for skill in self.skills:
            for mastery_level in self.mastery_levels:
                self.skills_attractiveness[
                    (skill, mastery_level)
                ] = self.get_skill_attractiveness(skill, mastery_level)

    def get_learner_attractiveness(self, learner):
        learner_attractiveness = 0
        for skill, mastery_level in learner["possessed_skills"].items():
            learner_attractiveness += self.skills_attractiveness[(skill, mastery_level)]
        return learner_attractiveness

    def get_all_learners_attractiveness(self):
        self.learners_attractiveness = Counter()
        for id_l, learner in self.learners.items():
            self.learners_attractiveness[id_l] = self.get_learner_attractiveness(
                learner
            )

    def get_avg_learner_attractiveness(self):
        avg_l_attrac = self.learners_attractiveness.total() / len(self.learners)
        return avg_l_attrac

    def get_nb_applicable_jobs(self, learner, threshold):
        nb_applicable_jobs = 0
        jobs_subset = set()
        for skill in learner["possessed_skills"]:
            if skill in self.jobs_inverted_index:
                jobs_subset.update(self.jobs_inverted_index[skill])
        for job_id in jobs_subset:
            matching = matchings.learner_job_matching(learner, self.jobs[job_id])
            if matching >= threshold:
                nb_applicable_jobs += 1
        return nb_applicable_jobs

    def get_avg_applicable_jobs(self, threshold):
        avg_applicable_jobs = 0
        for learner in self.learners.values():
            avg_applicable_jobs += self.get_nb_applicable_jobs(learner, threshold)
        avg_applicable_jobs /= len(self.learners)
        return avg_applicable_jobs

    def get_all_enrollable_courses(self, learner, threshold):
        enrollable_courses = {}
        for id_c, course in self.courses.items():
            required_matching = matchings.learner_course_required_matching(
                learner, course
            )
            provided_matching = matchings.learner_course_provided_matching(
                learner, course
            )
            if required_matching >= threshold and provided_matching < 1.0:
                enrollable_courses[id_c] = course
        return enrollable_courses

    def get_course_recommendation(self, learner, enrollable_courses, threshold):
        return recommendations.get_course_recommendation(
            learner,
            enrollable_courses,
            self.jobs,
            self.skills_attractiveness,
            threshold,
        )
