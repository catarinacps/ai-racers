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
EXP_ID=cmaes_ia_t1
# the code directory
CODE_DIR=$1
# the experiment directory
EXP_DIR=$2

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
cp -r $CODE_DIR code
mkdir results
results_csv=$(readlink -f results/${EXP_NAME}.csv)
weights_csv=$(readlink -f results/${EXP_NAME}_weights.csv)
pushd code

# init the csv results file
echo "sample_size,top_percentage,convergence_delta,iteration,bot,track,score" > $results_csv

# genetic algorithms
while read -r sample_size top_percentage convergence_delta; do
    iter=0
    csv_line=${sample_size},${top_percentage},${convergence_delta}

    for i in {1..6}; do
        # each plan has 16 combinations
        # therefore, we'll run 96 times

        while read -r bot track; do
            echo
            echo "--> Running with params: $sample_size $top_percentage $convergence_delta $bot $track"

            # run learning session
            python3 AIracers.py \
                    -t $track \
                    -b $bot \
                    -a ${sample_size},${top_percentage},${convergence_delta} \
                    -c 2 learn &> /dev/null

            score=$(grep 'Score:' cma_iter_w | awk '{print $2}')
            weights=$(grep 'Weights:' cma_iter_w | awk '{print $2}')

            # update iteration counter
            ((iter++))

            # commit results to csv
            echo ${csv_line},${iter},${bot},${track},${score} >> $results_csv
            echo ${weights} >> $weights_csv
        done < $EXP_DIR/runs.plan
    done
done < $EXP_DIR/cmaes.plan

popd

# pack everything and send to the exp dir
tar czf $EXP_DIR/data/$EXP_NAME.tar.gz *

popd
rm -rf $SCRATCH/*
