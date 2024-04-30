import json
import random

import pandas as pd
import numpy as np

from collections import defaultdict

import matchings


class Dataset:
    # The Dataset class is used to load and store the data of the recommendation problem
    def __init__(self, config):
        self.config = config
        self.load_data()
        self.get_jobs_inverted_index()

    def __str__(self):
        # override the __str__ method to print the dataset
        return (
            f"Dataset with {len(self.learners)} learners, "
            f"{len(self.jobs)} jobs, "
            f"{len(self.courses)} courses and "
            f"{len(self.skills)} skills."
        )

    def load_data(self):
        """Load the data from the files specified in the config and store it in the class attributes"""
        self.rng = random.Random(self.config["seed"])
        self.load_skills()
        self.load_mastery_levels()
        self.load_learners()
        self.load_jobs()
        self.load_courses()
        self.get_subsample()
        self.make_course_consistent()

    def load_skills(self):
        # load the skills from the taxonomy file
        self.skills = pd.read_csv(self.config["taxonomy_path"])

        # if level_3 is true, we only use the level 3 of the skill taxonomy, then we need to get the unique values in column Type Level 3
        if self.config["level_3"]:
            # get all the unique values in column Type Level 3
            level2int = {
                level: i for i, level in enumerate(self.skills["Type Level 3"].unique())
            }

            # make a dict from column unique_id to column Type Level 3
            skills_dict = dict(
                zip(self.skills["unique_id"], self.skills["Type Level 3"])
            )

            # map skills_dict values to level2int
            self.skills2int = {
                key: level2int[value] for key, value in skills_dict.items()
            }
            self.skills = set(self.skills2int.values())
        # if level_3 is false, we use the unique_id column as the skills
        else:
            self.skills = set(self.skills["unique_id"])
            self.skills2int = {skill: i for i, skill in enumerate(self.skills)}

    def load_mastery_levels(self):
        """Load the mastery levels from the file specified in the config and store it in the class attribute"""
        self.mastery_levels = json.load(open(self.config["mastery_levels_path"]))

    def get_avg_skills(self, skill_list, replace_unk):
        avg_skills = defaultdict(list)
        for skill, mastery_level in skill_list:
            # if the mastery level is a string and is in the mastery levels, we replace it with the corresponding value, otherwise we do nothing and continue to the next skill
            if isinstance(mastery_level, str) and mastery_level in self.mastery_levels:
                mastery_level = self.mastery_levels[mastery_level]
                if mastery_level == -1:
                    mastery_level = replace_unk
                skill = self.skills2int[skill]
                avg_skills[skill].append(mastery_level)
        # we take the average of the mastery levels for each skill because on our dataset we can have multiple mastery levels for the same skill
        for skill in avg_skills.keys():
            avg_skills[skill] = sum(avg_skills[skill]) / len(avg_skills[skill])
            avg_skills[skill] = round(avg_skills[skill])

        return avg_skills

    def load_learners(self, replace_unk=1):
        """Load the learners from the file specified in the config and store it in the class attribute

        Args:
            replace_unk (int, optional): The value to replace the unknown mastery levels. Defaults to 1.
        """
        learners = json.load(open(self.config["cv_path"]))
        self.max_learner_skills = self.config["max_cv_skills"]
        self.learners_index = dict()

        # numpy array to store the learners skill proficiency levels with default value 0
        self.learners = np.zeros((len(learners), len(self.skills)), dtype=int)
        index = 0

        # fill the numpy array with the learners skill proficiency levels from the json file
        for learner_id, learner in learners.items():

            avg_learner = self.get_avg_skills(learner, replace_unk)

            # if the number of skills is greater than the max_learner_skills, we skip the learner
            if len(avg_learner) > self.max_learner_skills:
                continue

            # we fill the numpy array with the averaged mastery levels
            for skill, level in avg_learner.items():
                self.learners[index][skill] = level

            self.learners_index[index] = learner_id
            self.learners_index[learner_id] = index

            index += 1

        # we update the learners numpy array with the correct number of rows
        self.learners = self.learners[:index]

    def load_jobs(self, replace_unk=3):
        """Load the jobs from the file specified in the config and store it in the class attribute

        Args:
            replace_unk (int, optional): The value to replace the unknown mastery levels. Defaults to 3.
        """
        jobs = json.load(open(self.config["job_path"]))
        self.jobs = np.zeros((len(jobs), len(self.skills)), dtype=int)
        self.jobs_index = dict()
        index = 0
        for job_id, job in jobs.items():
            self.jobs_index[index] = job_id
            self.jobs_index[job_id] = index
            avg_job = self.get_avg_skills(job, replace_unk)

            for skill, level in avg_job.items():
                self.jobs[index][skill] = level
            index += 1

    def load_courses(self, replace_unk=2):
        """Load the courses from the file specified in the config and store it in the class attribute

        Args:
            replace_unk (int, optional): The value to replace the unknown mastery levels. Defaults to 2.
        """
        courses = json.load(open(self.config["course_path"]))
        self.courses = np.zeros((len(courses), 2, len(self.skills)), dtype=int)
        self.courses_index = dict()
        index = 0
        for course_id, course in courses.items():
            # if the course does not provide any skills, we skip it
            if "to_acquire" not in course:
                continue

            self.courses_index[course_id] = index
            self.courses_index[index] = course_id

            avg_provided = self.get_avg_skills(course["to_acquire"], replace_unk)
            for skill, level in avg_provided.items():
                self.courses[index][1][skill] = level

            if "required" in course:
                avg_required = self.get_avg_skills(course["required"], replace_unk)
                for skill, level in avg_required.items():
                    self.courses[index][0][skill] = level

            index += 1
        # update the courses numpy array with the correct number of rows
        self.courses = self.courses[:index]

    def get_subsample(self):
        """Get a subsample of the dataset based on the config parameters"""
        random.seed(self.config["seed"])
        if self.config["nb_cvs"] != -1:
            # get a random sample of self.config["nb_cvs"] of ids from 0 to len(self.learners)
            learners_ids = random.sample(
                range(len(self.learners)), self.config["nb_cvs"]
            )
            # update the learners numpy array and the learners_index dictionary with the sampled ids
            self.learners = self.learners[learners_ids]
            self.learners_index = {
                i: self.learners_index[index] for i, index in enumerate(learners_ids)
            }
            self.learners_index.update({v: k for k, v in self.learners_index.items()})
        if self.config["nb_jobs"] != -1:
            jobs_ids = random.sample(range(len(self.jobs)), self.config["nb_jobs"])
            self.jobs = self.jobs[jobs_ids]
            self.jobs_index = {
                i: self.jobs_index[index] for i, index in enumerate(jobs_ids)
            }
            self.jobs_index.update({v: k for k, v in self.jobs_index.items()})
        if self.config["nb_courses"] != -1:
            courses_ids = random.sample(
                range(len(self.courses)), self.config["nb_courses"]
            )
            self.courses = self.courses[courses_ids]
            self.courses_index = {
                i: self.courses_index[index] for i, index in enumerate(courses_ids)
            }
            self.courses_index.update({v: k for k, v in self.courses_index.items()})

    def make_course_consistent(self):
        """Make the courses consistent by removing the skills that are provided and required at the same time"""
        for course in self.courses:
            for skill_id in range(len(self.skills)):
                required_level = course[0][skill_id]
                provided_level = course[1][skill_id]
                if provided_level != 0 and provided_level <= required_level:
                    if provided_level == 1:
                        course[0][skill_id] = 0
                    else:
                        course[0][skill_id] = provided_level - 1

    def get_jobs_inverted_index(self):
        """Get the inverted index for the jobs. The inverted index is a dictionary that maps the skill to the jobs that require it"""
        self.jobs_inverted_index = defaultdict(set)
        for i, job in enumerate(self.jobs):
            for skill, level in enumerate(job):
                if level > 0:
                    self.jobs_inverted_index[skill].add(i)

    def get_nb_applicable_jobs(self, learner, threshold):
        """Get the number of applicable jobs for a learner

        Args:
            learner (list): list of skills and mastery level of the learner
            threshold (float): the threshold for the matching

        Returns:
            int: the number of applicable jobs
        """
        nb_applicable_jobs = 0
        jobs_subset = set()

        # get the index of the non zero elements in the learner array
        skills = np.nonzero(learner)[0]

        for skill in skills:
            if skill in self.jobs_inverted_index:
                jobs_subset.update(self.jobs_inverted_index[skill])
        for job_id in jobs_subset:
            matching = matchings.learner_job_matching(learner, self.jobs[job_id])
            if matching >= threshold:
                nb_applicable_jobs += 1
        return nb_applicable_jobs

    def get_avg_applicable_jobs(self, threshold):
        """Get the average number of applicable jobs for all the learners

        Args:
            threshold (float): the threshold for the matching

        Returns:
            float: the average number of applicable jobs
        """
        avg_applicable_jobs = 0
        for learner in self.learners:
            avg_applicable_jobs += self.get_nb_applicable_jobs(learner, threshold)
        avg_applicable_jobs /= len(self.learners)
        return avg_applicable_jobs

    def get_all_enrollable_courses(self, learner, threshold):
        """Get all the enrollable courses for a learner

        Args:
            learner (list): list of skills and mastery level of the learner
            threshold (float): the threshold for the matching

        Returns:
            dict: dictionary of enrollable courses
        """
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
        """Get the attractiveness of a learner

        Args:
            learner (list): list of skills and mastery level of the learner

        Returns:
            int: number of jobs that require at least one of the learner's skills
        """
        attractiveness = 0

        # get the index of the non zero elements in the learner array
        skills = np.nonzero(learner)[0]

        for skill in skills:
            if skill in self.jobs_inverted_index:
                attractiveness += len(self.jobs_inverted_index[skill])
        return attractiveness

    def get_avg_learner_attractiveness(self):
        """Get the average attractiveness of all the learners

        Returns:
            float: the average attractiveness of the learners
        """
        attractiveness = 0
        for learner in self.learners:
            attractiveness += self.get_learner_attractiveness(learner)
        attractiveness /= len(self.learners)
        return attractiveness
