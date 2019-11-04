#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --chdir=.
#SBATCH --partition=tupi
#SBATCH --nodes=1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#BATCH --mail-type=END,FAIL
#BATCH --mail-user=hcpsilva@inf.ufrgs.br

set -euo pipefail

# machine:
MACHINE=${SLURMD_HOSTNAME}_${SLURM_CPUS_ON_NODE}

# parameters:
# the experiment ID, defined in the lab-book
EXP_ID=combined_ia_t1
# the experiment directory
EXP_DIR=$1

# experiment name (which is the ID and the machine and its core count)
EXP_NAME=${EXP_ID}_${MACHINE}

# go to the scratch dir
cd $SCRATCH

# and clean everything
rm -rf *

# prepare our directory
mkdir $EXP_NAME
pushd $EXP_NAME

# copy the code folder
cp -r $(dirname $EXP_DIR) code
mkdir results
results_csv=$(readlink -f results/${EXP_NAME}_combined.csv)
pushd code

# init the csv results file
echo "controller,neighbors,elitism,iteration,bot,track,score" > $results_csv

# combined run
while read -r controller neighbors elitism; do
    iter=0
    csv_line=${controller},${neighbors},${elitism}

    for i in {1..6}; do
        # each learning session will iterate 5 times
        # each plan has 16 combinations
        # therefore, we'll run 240 times

        while read -r bot track; do
            echo
            echo "--> Running with params: $controller $neighbors $elitism $bot $track"

            # run learning session
            score=$(python3 AIracers.py -t $track -b $bot -a $neighbors $elitism -c $controller learn)

            # update iteration counter
            ((iter+=5))

            # commit results to csv
            echo ${csv_line},${iter},${bot},${track},${score} >> $results_csv
        done < $EXP_DIR/runs.plan
    done
done < $EXP_DIR/combined.plan

popd

# pack everything and send to the exp dir
tar czf $EXP_DIR/data/$EXP_NAME.tar.gz *

popd
rm -rf $SCRATCH/*
