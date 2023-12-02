import os
import json

from copy import deepcopy

from Dataset import Dataset


class Greedy:
    def __init__(self, dataset, threshold):
        self.dataset = dataset
        self.threshold = threshold

    def update_learner_profile(self, learner, course):
        for skill, level in course["provided_skills"].items():
            if (
                skill not in learner["possessed_skills"]
                or learner["possessed_skills"][skill] < level
            ):
                learner["possessed_skills"][skill] = level

    def get_course_recommendation(self, learner, enrollable_courses):
        course_recommendation = None
        max_nb_applicable_jobs = 0
        max_attractiveness = 0

        for id_c, course in enrollable_courses.items():
            tmp_learner = deepcopy(learner)
            self.update_learner_profile(tmp_learner, course)

            nb_applicable_jobs = self.dataset.get_nb_applicable_jobs(
                tmp_learner, self.threshold
            )
            attractiveness = self.dataset.get_learner_attractiveness(tmp_learner)

            # Select the course that maximizes the number of applicable jobs

            if nb_applicable_jobs > max_nb_applicable_jobs:
                max_nb_applicable_jobs = nb_applicable_jobs
                course_recommendation = id_c
                max_attractiveness = attractiveness

            # If there are multiple courses that maximize the number of applicable jobs,
            # select the one that maximizes the attractiveness of the learner

            elif nb_applicable_jobs == max_nb_applicable_jobs:
                if attractiveness > max_attractiveness:
                    max_attractiveness = attractiveness
                    course_recommendation = id_c

        return course_recommendation

    def recommend_and_update(self, learner):
        enrollable_courses = self.dataset.get_all_enrollable_courses(
            learner, self.threshold
        )
        course_recommendation = self.get_course_recommendation(
            learner, enrollable_courses
        )

        self.update_learner_profile(
            learner, self.dataset.courses[course_recommendation]
        )
        return course_recommendation

    def greedy_recommendation(self, k):
        results = dict()

        avg_l_attrac = self.dataset.get_avg_learner_attractiveness()
        print(f"The average attractiveness of the learners is {avg_l_attrac:.2f}")

        results["original_attractiveness"] = avg_l_attrac

        avg_app_j = self.dataset.get_avg_applicable_jobs(self.threshold)
        print(f"The average nb of applicable jobs per learner is {avg_app_j:.2f}")

        results["original_applicable_jobs"] = avg_app_j

        recommendations = dict()

        for id_l in self.dataset.learners:
            recommendations[id_l] = []
            for i in range(k):
                recommendations[id_l].append(
                    self.recommend_and_update(self.dataset.learners[id_l])
                )

        new_learners_attractiveness = self.dataset.get_all_learners_attractiveness()

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
                    self.dataset.dataset_path, "results", "greedy_" + str(k) + ".json"
                ),
                "w",
            ),
            indent=4,
        )
