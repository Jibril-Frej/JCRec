# Skill Matching for Course Recommendation

## Installation

For now the installation requires [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html). Will add Docker later.

```shell script
git clone git@github.com:epfl-ml4ed/SkillThrills.git
cd SkillThrills
conda env create -f environment.yaml
conda activate proto
```

## Input

We have 4 raw files as input, for now they are json but this can be changed.

The files are in the folder data/raw/synthetic.

They can be generated using the script [protosp03/data/synthetic/make_synthetic.py](protosp03/data/synthetic/make_synthetic.py) and the config file [protosp03/config/synthetic.yaml](protosp03/config/synthetic.yaml)

- **skills.json** : a dict whoses keys are the skills name and values are the skill id
- **jobs.json** : a dict whoses keys are the job id and values are the list of skill required
- **resume.json** : a dict whoses keys are the resume id (or profile id) and values are the list of skill present
- **courses.json** : a dict whoses keys are the job id and values are the list of skill required and the list of skills provided

The [config](protosp03/config/synthetic.yaml) file provides requirements about the synthetic dataset (nb of skills, jobs, resume) and the path where the files are to be saved.  

To generate the synthetic data files:

```shell script
python protosp03/data/synthetic/make_synthetic.py --config protosp03/config/synthetic.yaml --seed 1
```

The default value for the seed is 42.

## Pre-Processing

Data preprocessing is done with the script [protosp03/data/synthetic/inverted_index.py](protosp03/data/synthetic/inverted_index.py) and the config file [protosp03/config/inverted_index.yaml](protosp03/config/inverted_index.yaml).

For now the main idea of the pre-processing is simply to create inverted indexes that can be used for effeicient search. The [config](protosp03/config/inverted_index.yaml) contains the path of the directory that contains the raw files and the path of the directory where to save the inverted indexes.

To pre-process the raw data files:  

```shell script
python protosp03/data/synthetic/inverted_index.py --config protosp03/config/inverted_index.yaml
```

## User-journey Version 1.0

### Running

To try the first version of the functions I can do in the user journey, run

```shell script
python -m tests.user_journey01 --seed 55
```

 It should display this:

```shell script
Considering Profile resume_2 with the skills:
        skill_20 at level 1 in group 4

We are assuming that the Profile is interested in Job jobs_6 that requires the skills:
        skill_9 at level 3 in group 1
        skill_10 at level 4 in group 0
        skill_6 at level 4 in group 3
        skill_21 at level 1 in group 0
        skill_20 at level 4 in group 4

The matching between the profile and the desired job is: 5%

The matching between the profile and the desired job for each group is
        Group 1 has a matching of 0%
        Group 0 has a matching of 0%
        Group 3 has a matching of 0%
        Group 4 has a matching of 25%

Printing the attractiveness of each skill of the profile and comparing to other learners:
        Skill skill_20 is required for 20% of the jobs on the market and 28% of the learners have it

The overall attractiveness of the profile is: 3%

Printing the matching of the profile with respect to each job (from most compatible to least compatible):
        Job jobs_5 has a matching of 33%
        Job jobs_6 has a matching of 20%
        Job jobs_7 has a matching of 16%
        Job jobs_13 has a matching of 14%
        Job jobs_19 has a matching of 11%

Printing the matching of the profile with respect to each course (from most compatible to least compatible):
        Course course_2 has a matching of 33%
                If the profile takes the course course_2, it will learn the new skills: {'skill_21'}
                The matching with the job jobs_6 will increase from 5% to: 25%
                The overall attractiveness of the profile will increase from 3 to: 8
```

## TODOs

Include time dimension
