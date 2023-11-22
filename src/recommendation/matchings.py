from collections import Counter
import sys
from copy import deepcopy


def skill_skill_similarity(provided_level, required_level):
    """
    Computes the similarity between two mastery levels of the same skill.

    Args:
        provided_level (int): Mastery level of the provided_skill.
        required_level (int): Mastery level of the required_skill.

    Returns:
        float: Similarity ratio for the given skill levels, ranging from 0 to 1.
    """
    return min(provided_level, required_level) / required_level


def learner_job_matching(learner, job):
    """
    Computes the compatibility score between a learner's skills and a job's required skills.

    For each required skill in the job, the function checks if the learner possesses that skill.
    If so, it calculates a similarity ratio based on the learner's mastery level and the
    job's required level for that skill. The final matching score is the average of all these
    similarity ratios for all required skills, expressed as a percentage.

    Args:
        learner (dict): Dictionary containing details about the learner.
                        - "possessed_skills": Dictionary where keys are skill names and values
                                              represent mastery levels.
                        - "year": Year associated with the learner's data.
        job (dict): Dictionary containing job requirements.
                    - "required_skills": Dictionary where keys are skill names and values
                                         represent required mastery levels.
                    - "year": Year associated with the job's data.

    Returns:
        float: Matching score between the learner and the job, ranging from 0 to 1.

    Example:
        learner = {
            "possessed_skills": {
                "Python": 3,
                "JavaScript": 1
            },
            "year": 2020
        }
        job = {
            "required_skills": {
        "Python": 2,
        "JavaScript": 3
            },
            "year": 2023
        }
        score = learner_job_matching(learner, job)
        print(score)  # This would output 66.66666
    """
    matching = 0

    # For each required skill in the job
    for skill in job["required_skills"]:
        # Check if the learner possesses the skill
        if skill in learner["possessed_skills"]:
            # Calculate similarity ratio based on mastery levels

            matching += skill_skill_similarity(
                learner["possessed_skills"][skill], job["required_skills"][skill]
            )

    matching = matching / len(job["required_skills"])

    return matching


def learner_course_required_matching(learner, course):
    """Computes the matching between a learner and a course based on the required skills.
    If the course has no required skills, the matching is 1.

    Args:
        learner (dict): Learner's profile including possessed skills and levels.
        course (dict): Course required and provided skills.

    Returns:
        float: matching value between 0 and 1
    """

    if not course["required_skills"]:
        return 1.0

    required_matching = 0
    for skill in course["required_skills"]:
        if skill in learner["possessed_skills"]:
            sim = skill_skill_similarity(
                learner["possessed_skills"][skill], course["required_skills"][skill]
            )
            required_matching += sim
    return required_matching / len(course["required_skills"])


def learner_course_provided_matching(learner, course):
    """Computes the matching between a learner and a course based on the provided skills.

    Args:
        learner (dict): Learner's profile including possessed skills and levels.
        course (dict): Course required and provided skills.

    Returns:
        float: matching value between 0 and 1
    """
    provided_matching = 0
    for skill in course["provided_skills"]:
        if skill in learner["possessed_skills"]:
            sim = skill_skill_similarity(
                learner["possessed_skills"][skill], course["provided_skills"][skill]
            )
            provided_matching += sim
    return provided_matching / len(course["provided_skills"])


def learner_course_matching(learner, course):
    """Computes the matching between a learner and a course.

    Args:
        learner (dict): Learner's profile including possessed skills and levels.
        course (dict): Course required and provided skills.

    Returns:
        float: matching value between 0 and 1
    """
    required_matching = learner_course_required_matching(learner, course)
    provided_matching = learner_course_provided_matching(learner, course)

    return required_matching * (1 - provided_matching)
