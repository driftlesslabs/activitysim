#!/usr/bin/env bash

WORKING_DIR=/tmp/edb-test

# make sure the working directory exists
mkdir -p $WORKING_DIR

# get the directory of this script
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# the path to the infer.py script
INFER_PY=${SCRIPT_DIR}/../scripts/infer.py

# activate the conda environment
echo The current CONDA environment is $CONDA_DEFAULT_ENV

# download the full model data
python ${SCRIPT_DIR}/download.py -d $WORKING_DIR

CONFIGS_DIR=${WORKING_DIR}/activitysim-prototype-mtc-extended/configs
FULL_DATA_DIR=${WORKING_DIR}/activitysim-prototype-mtc-extended/data_full
PSEUDOSURVEY_DIR=${WORKING_DIR}/activitysim-prototype-mtc-extended/output

# run activitysim to create outputs that will stand in for survey data
python -m activitysim run \
  -c ${CONFIGS_DIR} \
  -d ${FULL_DATA_DIR} \
  -o ${PSEUDOSURVEY_DIR} \
  --households_sample_size 200

# copy relevant outputs to where the infer.py script expects them
mkdir -p ${PSEUDOSURVEY_DIR}/survey_data
cp ${PSEUDOSURVEY_DIR}/final_*.csv ${PSEUDOSURVEY_DIR}/survey_data
cp ${PSEUDOSURVEY_DIR}/final_households.csv ${PSEUDOSURVEY_DIR}/survey_data/survey_households.csv
cp ${PSEUDOSURVEY_DIR}/final_persons.csv ${PSEUDOSURVEY_DIR}/survey_data/survey_persons.csv
cp ${PSEUDOSURVEY_DIR}/final_tours.csv ${PSEUDOSURVEY_DIR}/survey_data/survey_tours.csv
cp ${PSEUDOSURVEY_DIR}/final_joint_tour_participants.csv ${PSEUDOSURVEY_DIR}/survey_data/survey_joint_tour_participants.csv
cp ${PSEUDOSURVEY_DIR}/final_trips.csv ${PSEUDOSURVEY_DIR}/survey_data/survey_trips.csv

# create the output directory for the output of the infer.py script
OUTPUT_DIR=$WORKING_DIR/infer-output
mkdir -p $OUTPUT_DIR

# run the infer.py script
python $INFER_PY "${PSEUDOSURVEY_DIR}" "$CONFIGS_DIR" "$OUTPUT_DIR"

# copy the override files back into the data input directory
cp $OUTPUT_DIR/override_*.csv $FULL_DATA_DIR

# create a new output directory for the estimation run
EDB_DIR=${WORKING_DIR}/activitysim-prototype-mtc-extended/output-est-mode
mkdir -p ${EDB_DIR}

python -m activitysim run \
  -c ${SCRIPT_DIR}/../configs_estimation \
  -c ${CONFIGS_DIR} \
  -d ${FULL_DATA_DIR} \
  -o ${EDB_DIR}
