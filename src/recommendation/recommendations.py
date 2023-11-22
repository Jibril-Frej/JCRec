import matchings
import market
from copy import deepcopy


def get_course_recommendation(
    learner, enrollable_courses, jobs, skills_attractiveness, threshold
):
    """Get the course recommendation for a learner. The recommended course is chosen based on the
    learner's profile and the attractiveness of the skills that the course provides.

    Args:
        learner (dict): Learner's profile including possessed skills and levels.
        enrollable_courses (list): List of all enrollable courses.
        skills_attractiveness (Counter): attractiveness of all skills for each mastery level
    Returns:
        dict: The course recommendation
    """
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

        nb_applicable_jobs = market.get_nb_applicable_jobs(tmp_learner, jobs, threshold)
        attractiveness = market.get_learner_attractiveness(
            tmp_learner, skills_attractiveness
        )

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
