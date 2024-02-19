import os
import json

from copy import deepcopy
from time import time

# from Dataset import Dataset


class Optimal:
    def __init__(self, dataset, threshold):
        self.dataset = dataset
        self.threshold = threshold

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

    def update_learner_profile_list(self, learner, course_list):
        for id_c in course_list:
            self.update_learner_profile(learner, self.dataset.courses[id_c])

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
        if k == 0:
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

        else:
            enrollable_courses = self.dataset.get_all_enrollable_courses(
                learner, self.threshold
            )
            for id_c, course in enrollable_courses.items():
                tmp_learner = deepcopy(learner)
                self.update_learner_profile(tmp_learner, course)
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

    def recommend_and_update(self, learner, k):
        enrollable_courses = None
        candiate_course_recommendation_list = []
        course_recommendations_list = None
        max_nb_applicable_jobs = 0
        max_attractiveness = 0

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
        self.update_learner_profile_list(learner, course_recommendations_list)
        return course_recommendations_list

    def optimal_recommendation(self, k, run):
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
            recommendation_sequence = self.recommend_and_update(
                self.dataset.learners[i], k
            )

            recommendations[index] = [
                self.dataset.courses_index[course_id]
                for course_id in recommendation_sequence
            ]

        time_end = time()
        avg_recommendation_time = (time_end - time_start) / len(self.dataset.learners)

        print(f"Average Recommendation Time: {avg_recommendation_time:.2f} seconds")

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
