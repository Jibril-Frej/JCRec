import os
import json

from time import time

import numpy as np


class Optimal:
    def __init__(self, dataset, threshold):
        self.dataset = dataset
        self.threshold = threshold

    def update_learner_profile(self, learner, course):
        """Update the learner profile with the skills and levels provided by the course

        Args:
            learner (list): list of skills and mastery level of the learner
            course (list): list of required (resp. provided) skills and mastery level of the course
        """
        learner = np.maximum(learner, course[1])
        return learner

    def update_learner_profile_list(self, learner_id, course_list):
        """Update the learner profile with the skills and levels provided by all courses in the course list

        Args:
            learner_id (int): id of the learner
            course_list (list): list of courses
        """
        learner = self.dataset.learners[learner_id]
        for id_c in course_list:
            self.dataset.learners[learner_id] = self.update_learner_profile(
                learner, self.dataset.courses[id_c]
            )

    def get_course_recommendation(
        self,
        learner,
        enrollable_courses,
        candiate_course_recommendation_list,
        course_recommendations_list,
        max_nb_applicable_jobs,
        max_attractiveness,
        k,
    ):
        """Recursively get the optimal sequence of courses for a learner

        Args:
            learner (list): list of skills and mastery level of the learner
            enrollable_courses (dict): dictionary of courses that the learner can enroll in
            candiate_course_recommendation_list (list): list of candiate courses for the recommendation
            course_recommendations_list (list): optimal sequence of courses for the recommendation to be returned
            max_nb_applicable_jobs (int): current maximum number of applicable jobs given the recommendation list
            max_attractiveness (int): current maximum attractiveness given the recommendation list
            k (int): number of courses to recommend

        Returns:
            tuple: optimal sequence of courses for the recommendation, current maximum number of applicable jobs, current maximum attractiveness
        """
        if k == 0:
            # Base case: return the current recommendation list if the number of applicable jobs is greater than the current maximum
            nb_applicable_jobs = self.dataset.get_nb_applicable_jobs(
                learner, self.threshold
            )
            attractiveness = self.dataset.get_learner_attractiveness(learner)
            if nb_applicable_jobs > max_nb_applicable_jobs:
                course_recommendations_list = candiate_course_recommendation_list
                max_nb_applicable_jobs = nb_applicable_jobs
                max_attractiveness = attractiveness

            # If there are multiple courses that maximize the number of applicable jobs,
            # select the one that maximizes the attractiveness of the learner
            elif (
                nb_applicable_jobs == max_nb_applicable_jobs
                and attractiveness > max_attractiveness
            ):
                course_recommendations_list = candiate_course_recommendation_list
                max_nb_applicable_jobs = nb_applicable_jobs
                max_attractiveness = attractiveness

            return (
                course_recommendations_list,
                max_nb_applicable_jobs,
                max_attractiveness,
            )
        # Recursive case: for all courses that the learner can enroll in, get the optimal sequence of courses by calling the function recursively
        else:
            enrollable_courses = self.dataset.get_all_enrollable_courses(
                learner, self.threshold
            )
            for id_c, course in enrollable_courses.items():
                tmp_learner = learner
                tmp_learner = self.update_learner_profile(tmp_learner, course)
                new_candidate_list = candiate_course_recommendation_list + [id_c]
                (
                    course_recommendations_list,
                    max_nb_applicable_jobs,
                    max_attractiveness,
                ) = self.get_course_recommendation(
                    tmp_learner,
                    enrollable_courses,
                    new_candidate_list,
                    course_recommendations_list,
                    max_nb_applicable_jobs,
                    max_attractiveness,
                    k - 1,
                )

            return (
                course_recommendations_list,
                max_nb_applicable_jobs,
                max_attractiveness,
            )

    def recommend_and_update(self, learner_id, k):
        """Recommend a sequence of courses to the learner and update the learner profile

        Args:
            learner_id (int): id of the learner
            k (int): number of courses to recommend

        Returns:
            list: optimal sequence of courses for the recommendation
        """
        enrollable_courses = None
        candiate_course_recommendation_list = []
        course_recommendations_list = None
        max_nb_applicable_jobs = 0
        max_attractiveness = 0
        learner = self.dataset.learners[learner_id]

        (
            course_recommendations_list,
            max_nb_applicable_jobs,
            max_attractiveness,
        ) = self.get_course_recommendation(
            learner,
            enrollable_courses,
            candiate_course_recommendation_list,
            course_recommendations_list,
            max_nb_applicable_jobs,
            max_attractiveness,
            k,
        )
        self.update_learner_profile_list(learner_id, course_recommendations_list)
        return course_recommendations_list

    def optimal_recommendation(self, k, run):
        """Recommend a sequence of courses to all learners and save the results

        Args:
            k (int): number of courses to recommend
            run (int): run number
        """
        results = dict()

        avg_l_attrac = self.dataset.get_avg_learner_attractiveness()
        print(f"The average attractiveness of the learners is {avg_l_attrac:.2f}")

        results["original_attractiveness"] = avg_l_attrac

        avg_app_j = self.dataset.get_avg_applicable_jobs(self.threshold)
        print(f"The average nb of applicable jobs per learner is {avg_app_j:.2f}")

        results["original_applicable_jobs"] = avg_app_j

        time_start = time()
        recommendations = dict()
        for i, learner in enumerate(self.dataset.learners):
            index = self.dataset.learners_index[i]
            recommendation_sequence = self.recommend_and_update(i, k)

            recommendations[index] = [
                self.dataset.courses_index[course_id]
                for course_id in recommendation_sequence
            ]

        time_end = time()
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

        filename = (
            "optimal_nbskills_"
            + str(len(self.dataset.skills))
            + "_k_"
            + str(k)
            + "_run_"
            + str(run)
            + ".json"
        )

        json.dump(
            results,
            open(
                os.path.join(
                    self.dataset.config["results_path"],
                    filename,
                ),
                "w",
            ),
            indent=4,
        )
