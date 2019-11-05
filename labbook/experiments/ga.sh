#!/bin/bash


HOST=$(hostname)



# machine:

MACHINE=${HOST}



# parameters:

# the experiment ID, defined in the lab-book

EXP_ID=ga_ia_t1

# the code directory

CODE_DIR=$1

# the experiment directory

EXP_DIR=$CODE_DIR/labbook/experiments

# scratch

SCRATCH=$2



# experiment name (which is the ID and the machine and its core count)

EXP_NAME=${EXP_ID}_${MACHINE}



# go to the scratch dir

cd $SCRATCH



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

echo "population,mutation,elitism,iteration,bot,track,score" > $results_csv



# genetic algorithms

while read -r population mutation elitism; do

    iter=0

    csv_line=${population},${mutation},${elitism}



    for i in {1..4}; do

        # each plan has 16 combinations

        # therefore, we'll run 96 times



        while read -r bot track; do

            echo

            echo "--> Running with params: $population $mutation $elitism $bot $track"



            # run learning session
			python3 AIRacers.py -t $track -b $bot -a ${population},${mutation},${elitism} -c 1 learn &> /dev/null


            score=$(grep 'Score:' ga_iter_w | awk '{print $2}' | cat)

            weights=$(grep 'Weights:' ga_iter_w | awk '{print $2}' | cat)



            # update iteration counter

            iter=$((iter+1))



            # commit results to csvs

            echo ${csv_line},${iter},${bot},${track},${score} >> $results_csv

            echo ${weights} >> $weights_csv

        done < $EXP_DIR/runs.plan

    done



    # clean up current state so we start over again

    rm ga*.pkl

done < $EXP_DIR/ga.plan



popd



# pack everything and send to the exp dir

tar czf $EXP_DIR/data/$EXP_NAME.tar.gz *



popd
