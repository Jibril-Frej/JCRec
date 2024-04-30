import numpy as np


def matching(level1, level2):

    # get the minimum of the two arrays
    minimum_skill = np.minimum(level1, level2)

    # get the indices of the non zero elements of the job skill levels
    nonzero_indices = np.nonzero(level2)[0]

    # divide the minimum by the job skill levels on the non zero indices
    matching = minimum_skill[nonzero_indices] / level2[nonzero_indices]

    # sum the result and divide by the number of non zero job skill levels
    matching = np.sum(matching) / np.count_nonzero(level2)

    return matching


def learner_job_matching(learner, job):

    # check if one of the arrays is empty
    if not (np.any(job) and np.any(learner)):
        return 0

    return matching(learner, job)


def learner_course_required_matching(learner, course):

    required_course = course[0]

    # check if the course has no required skills and return 1
    if not np.any(required_course):
        return 1.0

    return matching(learner, required_course)


def learner_course_provided_matching(learner, course):

    provided_course = course[1]

    return matching(learner, provided_course)


def learner_course_matching(learner, course):

    # get the required and provided matchings
    required_matching = learner_course_required_matching(learner, course)
    provided_matching = learner_course_provided_matching(learner, course)

    return required_matching * (1 - provided_matching)
