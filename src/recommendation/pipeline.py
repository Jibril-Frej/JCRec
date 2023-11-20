import os
import json
import argparse
import logging

import market
import matchings
import upskillings
import recommendations


def get_dataset(dataset_path):
    filenames = [
        "skills.json",
        "mastery_levels.json",
        "years.json",
        "learners.json",
        "jobs.json",
        "courses.json",
    ]

    data = [json.load(open(os.path.join(dataset_path, fname))) for fname in filenames]

    skills, mastery_levels, years, learners, jobs, courses = data

    return skills, mastery_levels, years, learners, jobs, courses


def get_avg_learners_attractiveness(learners_attractiveness):
    return sum(learners_attractiveness) / len(learners_attractiveness)


def get_avg_applicable_jobs(learners, jobs, threshold):
    avg_applicable_jobs = 0
    for learner in learners:
        avg_applicable_jobs += matchings.get_nb_applicable_jobs(
            learner, jobs, threshold
        )
    avg_applicable_jobs /= len(learners)
    return avg_applicable_jobs


def greedy_recommendation(
    skills, mastery_levels, years, learners, jobs, courses, threshold, optimize, k=1
):
    (
        skill_supply,
        skill_demand,
        skill_trends,
        skills_attractiveness,
        learners_attractiveness,
    ) = market.get_all_market_metrics(skills, mastery_levels, learners, jobs, years)

    avg_learners_attractiveness = get_avg_learners_attractiveness(
        learners_attractiveness
    )

    logging.info(
        f"The average attractiveness of the learners is {avg_learners_attractiveness:.2f}"
    )

    avg_applicable_jobs = get_avg_applicable_jobs(learners, jobs, threshold)

    logging.info(
        f"The average number of applicable jobs per learner is {avg_applicable_jobs:.2f}"
    )

    no_recommendation = 0
    no_enrollable_courses = 0
    no_up_skilling_advice = 0
    no_learnable_skills = 0
    avg_nb_enrollable_courses = 0

    for learner in learners:
        # Get all enrollable courses for the learner

        enrollable_courses = matchings.get_all_enrollable_courses(
            learner, courses, threshold
        )

        if enrollable_courses is None:
            no_enrollable_courses += 1

        # Based on the enrollable courses, get all the learnable skills for the learner

        avg_nb_enrollable_courses += len(enrollable_courses)

        if len(learnable_skills) == 0:
            no_learnable_skills += 1

        # Among the learnable skills, get the up-skilling advice for the learner

        if optimize == "attractiveness":
            up_skilling_advice = upskillings.up_skilling_advice_attractiveness(
                learner, learnable_skills, skills_attractiveness
            )

        elif optimize == "applicability":
            up_skilling_advice = upskillings.up_skilling_advice_applicability(
                learner, learnable_skills, jobs, threshold
            )

        if up_skilling_advice is None:
            no_up_skilling_advice += 1

        course_recommendation = recommendations.get_course_recommendation(
            learner, enrollable_courses, up_skilling_advice
        )

        if course_recommendation is None:
            no_recommendation += 1

        else:
            for skill, level in course_recommendation["provided_skills"].items():
                if (
                    skill not in learner["possessed_skills"]
                    or learner["possessed_skills"][skill] < level
                ):
                    learner["possessed_skills"][skill] = level

    logging.debug(
        f"{no_enrollable_courses} learners ({no_enrollable_courses / len(learners) * 100:.2f}%) have no enrollable courses."
    )

    logging.debug(
        f"The average number of enrollable courses per learner is {avg_nb_enrollable_courses / len(learners):.2f}"
    )

    logging.debug(
        f"{no_learnable_skills} learners ({no_learnable_skills / len(learners) * 100:.2f}%) have no learnable skills."
    )

    logging.debug(
        f"{no_up_skilling_advice} learners ({no_up_skilling_advice / len(learners) * 100:.2f}%) have no advice."
    )

    logging.debug(
        f"{no_recommendation} learners ({no_recommendation / len(learners) * 100:.2f}%) have no course recommendation."
    )

    new_learners_attractiveness = market.get_all_learners_attractiveness(
        learners, skills_attractiveness
    )

    avg_learners_attractiveness = get_avg_learners_attractiveness(
        new_learners_attractiveness
    )

    logging.info(
        f"The new average attractiveness of the learners is {avg_learners_attractiveness:.2f}"
    )

    avg_applicable_jobs = get_avg_applicable_jobs(learners, jobs, threshold)

    logging.info(
        f"The new average number of applicable jobs per learner is {avg_applicable_jobs:.2f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path")
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument(
        "--optimize",
        type=str,
        default="attractiveness",
        choices=["attractiveness", "applicability"],
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    args = parser.parse_args()

    dataset_path = args.dataset_path
    threshold = args.threshold
    optimize = args.optimize
    log_level = args.log_level

    logging.basicConfig(level=log_level)

    skills, mastery_levels, years, learners, jobs, courses = get_dataset(dataset_path)

    print(f"\nOptimizing for attractiveness")

    greedy_recommendation(
        skills,
        mastery_levels,
        years,
        learners,
        jobs,
        courses,
        threshold,
        optimize,
    )


if __name__ == "__main__":
    main()
