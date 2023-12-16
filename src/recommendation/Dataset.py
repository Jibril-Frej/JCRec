import json
import os
import random

import yaml
import pandas as pd
from collections import Counter

import matchings


class Dataset:
    def __init__(self, config):
        self.config = config
        self.rng = None
        self.skills = None
        self.skills2int = None
        self.mastery_levels = None
        self.years = None
        self.learners_index = None
        self.learners = None
        self.jobs_index = None
        self.jobs = None
        self.jobs_inverted_index = None
        self.courses_index = None
        self.courses = None
        self.skill_supply = None
        self.skill_demand = None
        self.skills_attractiveness = None
        self.learners_attractiveness = None
        self.load_data()
        self.get_jobs_inverted_index()

    def __str__(self):
        return (
            f"Dataset with {len(self.learners)} learners, "
            f"{len(self.jobs)} jobs, "
            f"{len(self.courses)} courses and "
            f"{len(self.skills)} skills."
        )

    def load_data(self):
        with open(self.config, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        self.rng = random.Random(self.config["seed"])
        self.skills = pd.read_csv(self.config["taxonomy_path"])

        # get all the unique values in column Type Level 3
        level2int = {
            level: i for i, level in enumerate(self.skills["Type Level 3"].unique())
        }

        # make a dict from column unique_id to column Type Level 3
        skills_dict = dict(zip(self.skills["unique_id"], self.skills["Type Level 3"]))

        # map skills_dict values to level2int
        self.skills2int = {key: level2int[value] for key, value in skills_dict.items()}
        self.skills = set(self.skills2int.values())

        self.mastery_levels = json.load(open(self.config["mastery_levels_path"]))
        self.load_learners()
        self.load_jobs()
        self.load_courses()
        self.get_subsample()
        self.make_course_consistent()
        self.make_indexes()

    def load_learners(self, replace_unk=1):
        learners = json.load(open(self.config["cv_path"]))
        self.learners_index = dict()
        index = 0
        self.learners = dict()
        for learner_id, learner in learners.items():
            self.learners[learner_id] = dict()
            self.learners_index[learner_id] = index
            self.learners_index[index] = learner_id
            for skill, mastery_level in learner:
                if (
                    isinstance(mastery_level, str)
                    and mastery_level in self.mastery_levels
                ):
                    mastery_level = self.mastery_levels[mastery_level]
                    if mastery_level == -1:
                        mastery_level = replace_unk
                    skill = self.skills2int[skill]
                    if skill not in self.learners[learner_id]:
                        self.learners[learner_id][skill] = []
                    self.learners[learner_id][skill].append(mastery_level)
            for skill, level_list in self.learners[learner_id].items():
                self.learners[learner_id][skill] = round(
                    sum(level_list) / len(level_list)
                )

    def load_jobs(self, replace_unk=3):
        jobs = json.load(open(self.config["job_path"]))
        self.jobs = dict()
        for job_id, job in jobs.items():
            self.jobs[job_id] = dict()
            for skill, mastery_level in job:
                if (
                    isinstance(mastery_level, str)
                    and mastery_level in self.mastery_levels
                ):
                    mastery_level = self.mastery_levels[mastery_level]
                    if mastery_level == -1:
                        mastery_level = replace_unk
                    skill = self.skills2int[skill]
                    if skill not in self.jobs[job_id]:
                        self.jobs[job_id][skill] = []
                    self.jobs[job_id][skill].append(mastery_level)
            for skill, level_list in self.jobs[job_id].items():
                self.jobs[job_id][skill] = round(sum(level_list) / len(level_list))

    def load_courses(self, replace_unk=2):
        courses = json.load(open(self.config["course_path"]))
        self.courses = dict()
        for course_id, course in courses.items():
            self.courses[course_id] = {
                "required_skills": dict(),
                "provided_skills": dict(),
            }
            if "required" in course:
                for skill, mastery_level in course["required"]:
                    if (
                        isinstance(mastery_level, str)
                        and mastery_level in self.mastery_levels
                    ):
                        mastery_level = self.mastery_levels[mastery_level]
                        if mastery_level == -1:
                            mastery_level = replace_unk
                        skill = self.skills2int[skill]
                        if skill not in self.courses[course_id]["required_skills"]:
                            self.courses[course_id]["required_skills"][skill] = []
                        self.courses[course_id]["required_skills"][skill].append(
                            mastery_level
                        )
                for skill, level_list in self.courses[course_id][
                    "required_skills"
                ].items():
                    self.courses[course_id]["required_skills"][skill] = round(
                        sum(level_list) / len(level_list)
                    )
            if "to_acquire" in course:
                for skill, mastery_level in course["to_acquire"]:
                    if (
                        isinstance(mastery_level, str)
                        and mastery_level in self.mastery_levels
                    ):
                        mastery_level = self.mastery_levels[mastery_level]
                        if mastery_level == -1:
                            mastery_level = replace_unk
                        skill = self.skills2int[skill]
                        if skill not in self.courses[course_id]["provided_skills"]:
                            self.courses[course_id]["provided_skills"][skill] = []
                        self.courses[course_id]["provided_skills"][skill].append(
                            mastery_level
                        )
                for skill, level_list in self.courses[course_id][
                    "provided_skills"
                ].items():
                    self.courses[course_id]["provided_skills"][skill] = round(
                        sum(level_list) / len(level_list)
                    )

    def get_subsample(self):
        random.seed(self.config["seed"])
        if self.config["nb_cvs"] != -1:
            self.learners = dict(
                self.rng.sample(self.learners.items(), self.config["nb_cvs"])
            )
        if self.config["nb_jobs"] != -1:
            self.jobs = dict(self.rng.sample(self.jobs.items(), self.config["nb_jobs"]))
        if self.config["nb_courses"] != -1:
            self.courses = dict(
                self.rng.sample(self.courses.items(), self.config["nb_courses"])
            )

    def make_course_consistent(self):
        courses_to_remove = []
        for course_id, course in self.courses.items():
            for skill in course["provided_skills"]:
                if skill in course["required_skills"]:
                    required_level = course["required_skills"][skill]
                    provided_level = course["provided_skills"][skill]
                    if provided_level <= required_level:
                        if provided_level == 1:
                            course["required_skills"].pop(skill)
                        else:
                            course["required_skills"][skill] = provided_level - 1
            if not course["provided_skills"]:
                courses_to_remove.append(course_id)
        for course_id in courses_to_remove:
            self.courses.pop(course_id)

    def make_indexes(self):
        self.make_learners_index()
        self.make_jobs_index()
        self.make_courses_index()

    def make_learners_index(self):
        self.learners_index = dict()
        index = 0
        tmp_learners = []
        for learner_id, learner in self.learners.items():
            self.learners_index[learner_id] = index
            self.learners_index[index] = learner_id
            tmp_learners.append([(skill, level) for skill, level in learner.items()])
            index += 1
        self.learners = tmp_learners

    def make_jobs_index(self):
        self.jobs_index = dict()
        index = 0
        tmp_jobs = []
        for job_id, job in self.jobs.items():
            self.jobs_index[job_id] = index
            self.jobs_index[index] = job_id
            tmp_jobs.append([(skill, level) for skill, level in job.items()])
            index += 1
        self.jobs = tmp_jobs

    def make_courses_index(self):
        self.courses_index = dict()
        index = 0
        tmp_courses = []
        for course_id, course in self.courses.items():
            self.courses_index[course_id] = index
            self.courses_index[index] = course_id
            tmp_course = [[], []]
            for skill, level in course["required_skills"].items():
                tmp_course[0].append((skill, level))
            for skill, level in course["provided_skills"].items():
                tmp_course[1].append((skill, level))
            tmp_courses.append(tmp_course)
            index += 1
        self.courses = tmp_courses

    def get_jobs_inverted_index(self):
        self.jobs_inverted_index = dict()
        for i, job in enumerate(self.jobs):
            for skill, level in job:
                if skill not in self.jobs_inverted_index:
                    self.jobs_inverted_index[skill] = set()
                self.jobs_inverted_index[skill].add(i)

    def get_nb_applicable_jobs(self, learner, threshold):
        nb_applicable_jobs = 0
        jobs_subset = set()
        for skill, level in learner:
            if skill in self.jobs_inverted_index:
                jobs_subset.update(self.jobs_inverted_index[skill])
        for job_id in jobs_subset:
            matching = matchings.learner_job_matching(learner, self.jobs[job_id])
            if matching >= threshold:
                nb_applicable_jobs += 1
        return nb_applicable_jobs

    def get_avg_applicable_jobs(self, threshold):
        avg_applicable_jobs = 0
        for learner in self.learners:
            avg_applicable_jobs += self.get_nb_applicable_jobs(learner, threshold)
        avg_applicable_jobs /= len(self.learners)
        return avg_applicable_jobs

    def get_all_enrollable_courses(self, learner, threshold):
        enrollable_courses = {}
        for i, course in enumerate(self.courses):
            required_matching = matchings.learner_course_required_matching(
                learner, course
            )
            provided_matching = matchings.learner_course_provided_matching(
                learner, course
            )
            if required_matching >= threshold and provided_matching < 1.0:
                enrollable_courses[i] = course
        return enrollable_courses

    def get_learner_attractiveness(self, learner):
        attractiveness = 0
        for skill, level in learner:
            attractiveness += len(self.jobs_inverted_index[skill])
        return attractiveness

    def get_avg_learner_attractiveness(self):
        attractiveness = 0
        for learner in self.learners:
            attractiveness += self.get_learner_attractiveness(learner)
        attractiveness /= len(self.learners)
        return attractiveness
