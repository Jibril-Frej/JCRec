import matchings
import market


def get_course_recommendation(learner, enrollable_courses, skills_attractiveness):
    """Get the course recommendation for a learner and a given up-skilling advice.

    Args:
        learner (dict): Learner's profile including possessed skills and levels.
        enrollable_courses (list): List of all enrollable courses.
        up_skilling_advice (tuple): A tuple of the skill and level to up-skill to.

    Returns:
        dict: The course recommendation
    """
    course_recommendation = None

    max_attractiveness = 0
    for course in enrollable_courses:
        tmp_learner = deepcopy(learner)
        for skill, level in course["provided_skills"].items():
            if (
                skill not in tmp_learner["possessed_skills"]
                or tmp_learner["possessed_skills"][skill] < level
            ):
                tmp_learner["possessed_skills"][skill] = level

    return course_recommendation
