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
EXP_ID=hc_ia_t1
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
results_csv=$(readlink -f results/${EXP_NAME}.csv)
weights_csv=$(readlink -f results/${EXP_NAME}_weights.csv)
pushd code

# init the csv results file
echo "percentage,iteration,bot,track,score" > $results_csv

# genetic algorithms
while read -r percentage pop; do
    iter=0
    csv_line=${percentage}

    for i in {1..6}; do
        # each plan has 16 combinations
        # therefore, we'll run 96 times

        while read -r bot track; do
            echo
            echo "--> Running with params: $percentage $bot $track"

            # run learning session
            python3 AIracers.py \
                    -t $track \
                    -b $bot \
                    -a ${percentage} \
                    -c 3 learn &> /dev/null

            score=$(grep 'Score:' hc_iter_w | awk '{print $2}')
            weights=$(grep 'Weights:' hc_iter_w | awk '{print $2}')

            # update iteration counter
            ((iter++))

            # commit results to csv
            echo ${csv_line},${iter},${bot},${track},${score} >> $results_csv
            echo ${weights} >> $weights_csv
        done < $EXP_DIR/runs.plan
    done
done < $EXP_DIR/hc.plan

popd

# pack everything and send to the exp dir
tar czf $EXP_DIR/data/$EXP_NAME.tar.gz *

popd
rm -rf $SCRATCH/*
