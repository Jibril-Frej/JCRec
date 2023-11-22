from copy import deepcopy

from Dataset import Dataset


# import recommendations


class Greedy:
    def __init__(self, dataset, threshold):
        self.dataset = dataset
        self.threshold = threshold

    def get_course_recommendation(self, learner, enrollable_courses):
        course_recommendation = None
        max_nb_applicable_jobs = 0
        max_attractiveness = 0

        for id_c, course in enrollable_courses.items():
            tmp_learner = deepcopy(learner)
            for skill, level in course["provided_skills"].items():
                if (
                    skill not in tmp_learner["possessed_skills"]
                    or tmp_learner["possessed_skills"][skill] < level
                ):
                    tmp_learner["possessed_skills"][skill] = level

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

    def greedy_recommendation(self, k):
        avg_l_attrac = self.dataset.get_avg_learner_attractiveness()
        print(f"The average attractiveness of the learners is {avg_l_attrac:.2f}")

        avg_app_j = self.dataset.get_avg_applicable_jobs(self.threshold)
        print(f"The average nb of applicable jobs per learner is {avg_app_j:.2f}")

        no_enrollable_courses = 0
        avg_nb_enrollable_courses = 0

        for id_l in self.dataset.learners:
            for i in range(k):
                enrollable_courses = self.dataset.get_all_enrollable_courses(
                    self.dataset.learners[id_l], self.threshold
                )

                if enrollable_courses is None:
                    no_enrollable_courses += 1

                avg_nb_enrollable_courses += len(enrollable_courses)

                course_recommendation = self.get_course_recommendation(
                    self.dataset.learners[id_l], enrollable_courses
                )

                for skill, level in self.dataset.courses[course_recommendation][
                    "provided_skills"
                ].items():
                    if (
                        skill not in self.dataset.learners[id_l]["possessed_skills"]
                        or self.dataset.learners[id_l]["possessed_skills"][skill]
                        < level
                    ):
                        self.dataset.learners[id_l]["possessed_skills"][skill] = level

        new_learners_attractiveness = self.dataset.get_all_learners_attractiveness()

        avg_l_attrac = self.dataset.get_avg_learner_attractiveness()
        print(f"The new average attractiveness of the learners is {avg_l_attrac:.2f}")

        avg_app_j = self.dataset.get_avg_applicable_jobs(self.threshold)
        print(f"The new average nb of applicable jobs per learner is {avg_app_j:.2f}")