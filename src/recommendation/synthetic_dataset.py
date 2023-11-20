import os
import random
import yaml
import argparse
import json

import pandas as pd
import numpy as np


def read_taxonomy(path):
    """Reads the taxonomy from a csv file, keeps only the essential columns and
    returns the taxonomy as a pandas dataframe.

    Args:
        path (str): path to the csv file

    Returns:
        dataframe: the taxonomy as a pandas dataframe
    """
    taxonomy = pd.read_csv(path)

    # remove all rows where the column 'ElementID' is null
    taxonomy = taxonomy[taxonomy["ElementID"].notna()]

    keep = [
        "ElementID",
        "Dimension FE",
        "Type Level 1",
        "Type Level 1 E",
        "Type Level 2",
        "Type Level 2 E",
        "Type Level 3",
        "Type Level 4",
    ]

    # keep only the columns in the list 'keep'
    taxonomy = taxonomy[keep]
    names = ["Type Level 1", "Type Level 2", "Type Level 3", "Type Level 4"]

    taxonomy["last_name"] = (taxonomy["ElementID"].str.len() - 1) // 2
    taxonomy["last_name"] = taxonomy["last_name"].map(lambda x: names[x])
    taxonomy["last_name"] = taxonomy.apply(lambda x: x[x["last_name"]], axis=1)
    return taxonomy


def get_mastery_levels_proba(mastery_levels):
    """Returns a probability distribution over the mastery levels.

    Args:
        mastery_levels (list): list of mastery levels

    Returns:
        array: a probability distribution over the mastery levels
    """
    nb_mastery_levels = len(mastery_levels)
    mastery_levels_probabilities = [
        1 / np.log(i + 1) for i in range(1, nb_mastery_levels + 1)
    ]
    mastery_levels_normalized_probabilities = np.array(
        mastery_levels_probabilities
    ) / sum(mastery_levels_probabilities)
    return mastery_levels_normalized_probabilities


def get_skills(taxonomy):
    """Returns a list of skills and a probability distribution over the skills.

    Args:
        taxonomy (dataframe): the taxonomy

    Returns:
        list, array: a list of skills and a probability distribution over the skills
    """

    levels_dict = taxonomy.set_index("last_name")["ElementID"].to_dict()
    levels_dict = {
        key: [int(level) for level in value.split(".")]
        for key, value in levels_dict.items()
    }

    skills = list(levels_dict.keys())
    random.shuffle(skills)
    nb_skills = len(skills)
    skills_probabilities = [1 / np.log(i + 1) for i in range(1, nb_skills + 1)]
    skills_normalized_probabilities = np.array(skills_probabilities) / sum(
        skills_probabilities
    )
    return skills, skills_normalized_probabilities


def get_years_proba(years):
    """Returns a probability distribution over the years.

    Args:
        years (list): list of years

    Returns:
        array: a probability distribution over the years
    """
    years_probabilities = [1 / np.log(i + 1) for i in range(1, len(years) + 1)]
    years_normalized_probabilities = np.array(years_probabilities) / sum(
        years_probabilities
    )
    return years_normalized_probabilities


def get_random_learner(
    skills,
    mastery_levels,
    years,
    skills_normalized_probabilities,
    mastery_levels_normalized_probabilities,
    years_normalized_probabilities,
    min_n_skills=5,
    max_n_skills=10,
):
    """Creates a random learner with a random number of skills and mastery levels and a random year.

    Args:
        skills (list): list of skills
        mastery_levels (list): list of mastery levels
        years (list): list of years
        skills_normalized_probabilities (array): a probability distribution over the skills
        mastery_levels_normalized_probabilities (array): a probability distribution over the skills
        years_normalized_probabilities (array): a probability distribution over the skills
        min_n_skills (int, optional): minimum number of skills that a user has. Defaults to 5.
        max_n_skills (int, optional): maximum number of skills that a user has. Defaults to 10.

    Returns:
        dict: a dictionary containing the (skills,mastery levels) and the year of the learner
    """
    n_skills = random.randint(min_n_skills, max_n_skills)
    possessed = {
        skill: level.item()
        for skill, level in zip(
            np.random.choice(
                skills, n_skills, p=skills_normalized_probabilities, replace=False
            ),
            np.random.choice(
                mastery_levels,
                n_skills,
                p=mastery_levels_normalized_probabilities,
                replace=True,
            ),
        )
    }
    year = np.random.choice(years, 1, p=years_normalized_probabilities)[0].item()
    return {"possessed_skills": possessed, "year": year}


def get_all_learners(
    skills,
    mastery_levels,
    years,
    skills_normalized_probabilities,
    mastery_levels_normalized_probabilities,
    years_normalized_probabilities,
    min_n_skills=5,
    max_n_skills=10,
    n_learners=100,
):
    """Creates a list of random learners.

    Args:
        skills (list): list of skills
        mastery_levels (list): list of mastery levels
        years (list): list of yearsyears
        skills_normalized_probabilities (array): a probability distribution over the skills
        mastery_levels_normalized_probabilities (array): a probability distribution over the skills
        years_normalized_probabilities (array): a probability distribution over the skills
        min_n_skills (int, optional): minimum number of skills that a user has. Defaults to 5.
        max_n_skills (int, optional): maximum number of skills that a user has. Defaults to 10.
        n_learners (int, optional): number of learners to create. Defaults to 100.

    Returns:
        list: a list of random learners
    """
    return [
        get_random_learner(
            skills,
            mastery_levels,
            years,
            skills_normalized_probabilities,
            mastery_levels_normalized_probabilities,
            years_normalized_probabilities,
            min_n_skills,
            max_n_skills,
        )
        for _ in range(n_learners)
    ]


def get_random_job(
    skills,
    mastery_levels,
    years,
    skills_normalized_probabilities,
    mastery_levels_normalized_probabilities,
    years_normalized_probabilities,
    min_n_skills=2,
    max_n_skills=5,
):
    """Creates a random job with a random number of skills and mastery levels and a random year.

    Args:
        skills (list): list of skills
        mastery_levels (list): list of mastery levels
        years (list): list of years
        skills_normalized_probabilities (array): a probability distribution over the skills
        mastery_levels_normalized_probabilities (array): a probability distribution over the skills
        years_normalized_probabilities (array): a probability distribution over the skills
        min_n_skills (int, optional): . Defaults to 2.
        max_n_skills (int, optional): . Defaults to 5.

    Returns:
        dict: a dictionary containing the (skills,mastery levels) and the year of the job
    """
    n_skills = random.randint(min_n_skills, max_n_skills)
    required = {
        skill: level.item()
        for skill, level in zip(
            np.random.choice(
                skills, n_skills, p=skills_normalized_probabilities, replace=False
            ),
            np.random.choice(
                mastery_levels,
                n_skills,
                p=mastery_levels_normalized_probabilities,
                replace=True,
            ),
        )
    }
    year = np.random.choice(years, 1, p=years_normalized_probabilities)[0]
    return {"required_skills": required, "year": year.item()}


def get_all_jobs(
    skills,
    mastery_levels,
    years,
    skills_normalized_probabilities,
    mastery_levels_normalized_probabilities,
    years_normalized_probabilities,
    min_n_skills=2,
    max_n_skills=5,
    n_jobs=1000,
):
    """Creates a list of random jobs.

    Args:
        skills (list): list of skills
        mastery_levels (list): list of mastery levels
        years (list): list of years
        skills_normalized_probabilities (array): a probability distribution over the skills
        mastery_levels_normalized_probabilities (array): a probability distribution over the skills
        years_normalized_probabilities (array): a probability distribution over the skills
        min_n_skills (int, optional): minimum number of skills that a job requires. Defaults to 2.
        max_n_skills (int, optional): maximum number of skills that a job requires. Defaults to 5.
        n_jobs (int, optional): number of jobs to create. Defaults to 100.

    Returns:
        dict: a dictionary containing the (skills,mastery levels) and the year of the job
    """

    return [
        get_random_job(
            skills,
            mastery_levels,
            years,
            skills_normalized_probabilities,
            mastery_levels_normalized_probabilities,
            years_normalized_probabilities,
            min_n_skills,
            max_n_skills,
        )
        for _ in range(n_jobs)
    ]


def get_random_provided_skills(
    skills, mastery_levels, required_skills, n_provided_skills
):
    """Returns a dictionary of provided skills for a course.

    Args:
        skills (list): list of skills
        mastery_levels (list): list of mastery levels
        required_skills (dict): dictionary of required skills
        n_provided_skills (int): number of provided skills

    Returns:
        dict: dictionary of provided skills
    """
    provided_skills = dict()
    while len(provided_skills) < n_provided_skills:
        candidate_skill = random.choice(skills)
        candidate_level = random.choice(mastery_levels)
        if (
            candidate_skill not in required_skills
            and candidate_skill not in provided_skills
        ):
            provided_skills[candidate_skill] = candidate_level
        elif (
            candidate_skill in required_skills
            and candidate_level > required_skills[candidate_skill]
        ):
            provided_skills[candidate_skill] = candidate_level

    return provided_skills


def get_random_course(
    skills,
    mastery_levels,
    min_n_required_skills=1,
    max_n_required_skills=5,
    min_n_provided_skills=1,
    max_n_provided_skills=2,
):
    """Creates a random course with a random number of required and provided skills.

    Args:
        skills (list): list of skills
        mastery_levels (list): list of mastery levels
        min_n_required_skills (int, optional): minimum number of required skills. Defaults to 1.
        max_n_required_skills (int, optional): maximum number of required skills. Defaults to 5.
        min_n_provided_skills (int, optional): minimum number of provided skills. Defaults to 1.
        max_n_provided_skills (int, optional): maximum number of provided skills. Defaults to 2.

    Returns:
        dict: a dictionary containing the required and provided skills of the course
    """
    n_required_skills = random.randint(min_n_required_skills, max_n_required_skills)
    required = {
        skill: level.item()
        for skill, level in zip(
            np.random.choice(skills, n_required_skills, replace=False),
            np.random.choice(mastery_levels, n_required_skills, replace=True),
        )
    }

    n_provided_skills = random.randint(min_n_provided_skills, max_n_provided_skills)
    provided = get_random_provided_skills(
        skills, mastery_levels, required, n_provided_skills
    )

    return {"required_skills": required, "provided_skills": provided}


def get_all_courses(
    skills,
    mastery_levels,
    min_n_required_skills=1,
    max_n_required_skills=5,
    min_n_provided_skills=1,
    max_n_provided_skills=2,
    n_courses=1000,
):
    """

    Args:
        skills (_type_): _description_
        mastery_levels (_type_): _description_
        min_n_required_skills (int, optional): _description_. Defaults to 1.
        max_n_required_skills (int, optional): _description_. Defaults to 5.
        min_n_provided_skills (int, optional): _description_. Defaults to 1.
        max_n_provided_skills (int, optional): _description_. Defaults to 2.
        n_courses (int, optional): _description_. Defaults to 1000.

    Returns:
        _type_: _description_
    """
    return [
        get_random_course(
            skills,
            mastery_levels,
            min_n_required_skills,
            max_n_required_skills,
            min_n_provided_skills,
            max_n_provided_skills,
        )
        for _ in range(n_courses)
    ]


def get_job_market(
    taxonomy_path="../data/taxonomy/taxonomy_V4.csv",
    mastery_levels=[1, 2, 3, 4],
    years=[i for i in range(2023, 2017, -1)],
    learner_min_n_skills=5,
    learner_max_n_skills=10,
    n_learners=1000,
    job_min_n_skills=2,
    job_max_n_skills=5,
    job_n_jobs=1000,
    course_min_n_required_skills=1,
    course_max_n_required_skills=5,
    course_min_n_provided_skills=1,
    course_max_n_provided_skills=2,
    n_courses=1000,
):
    """Creates a job market with random learners and jobs.

    Args:
        taxonomy_path (str, optional): path of the taxonomy. Defaults to "../data/taxonomy/taxonomy_V4.csv".
        mastery_levels (list, optional): list of mastery levels. Defaults to [1, 2, 3, 4].
        years (list, optional): list of years. Defaults to [i for i in range(2023, 2017, -1)].

    Returns:
        list, list, list: a list of skills, a list of learners and a list of jobs
    """
    taxonomy = read_taxonomy(taxonomy_path)
    mastery_levels_normalized_probabilities = get_mastery_levels_proba(mastery_levels)
    skills, skills_normalized_probabilities = get_skills(taxonomy)
    years_normalized_probabilities = get_years_proba(years)
    learners = get_all_learners(
        skills,
        mastery_levels,
        years,
        skills_normalized_probabilities,
        mastery_levels_normalized_probabilities,
        years_normalized_probabilities,
        learner_min_n_skills,
        learner_max_n_skills,
        n_learners,
    )

    jobs = get_all_jobs(
        skills,
        mastery_levels,
        years,
        skills_normalized_probabilities,
        mastery_levels_normalized_probabilities,
        years_normalized_probabilities,
        job_min_n_skills,
        job_max_n_skills,
        job_n_jobs,
    )

    courses = get_all_courses(
        skills,
        mastery_levels,
        course_min_n_required_skills,
        course_max_n_required_skills,
        course_min_n_provided_skills,
        course_max_n_provided_skills,
        n_courses,
    )

    return skills, learners, jobs, courses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()

    config = args.config
    with open(config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    dataset_path = config.pop("dataset_path")

    seed = config.pop("seed")
    random.seed(seed)

    skills, learners, jobs, courses = get_job_market(**config)

    data_to_save = {
        "skills.json": skills,
        "mastery_levels.json": config["mastery_levels"],
        "years.json": config["years"],
        "learners.json": learners,
        "jobs.json": jobs,
        "courses.json": courses,
    }

    for json_file, data in data_to_save.items():
        with open(os.path.join(dataset_path, json_file), "w") as f:
            json.dump(data, f, indent=4)


if __name__ == "__main__":
    main()
