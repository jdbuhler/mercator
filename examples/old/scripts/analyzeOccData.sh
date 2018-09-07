#!/bin/bash 

# AWK script to analyze the occupancy instrumentation output of MERCATOR.
# Expected format of occ rows in CSV file:
# blockID, instrumentation ID (value currently 3 for OCC), module ID, numItems, \
#   numFirings, numFullFirings, occRatio, fullRatio
#
# where: 
#  - numItems is total items executed by a module.
#  - a fullFiring is a firing on a number of items equal to the max fired in
#    one chunk, which is the lesser of the user-imposed maxInputs and the block
#    width in threads. Example: 128 for all experiments in hpcs2017 paper.
#  - occRatio is numItems/numFirings.
#  - fullRatio is numFullFirings/numFirings.

# Goal: average statistics across blocks for each module.

MAX_FIRING_WIDTH=128

usage="Usage: $0 <occ results directory>    NB: Results assumed to be generated
by MERCATOR with INSTRUMENT_OCC set to 1."

# must have src dir with results files
if [ "$#" -ne 1 ];
then
  echo ${usage}
  exit 1
fi

# check whether src dir exists
srcDir=$1
if ! [ -e ${srcDir} ];
then
  echo "Error: directory ${srcDir} not found."
  exit 1
fi

# check whether src dir is a dir
if ! [ -d ${srcDir} ];
then
  echo "Error: ${srcDir} is not a directory."
  exit 1
fi

## recursively sort all files in src dir; place analyzed filed in destination dir
dstDir=${srcDir}_agg_occ

mkdir -p ${dstDir}
# for each app (assuming each has its own folder in srcDir)
for topoDir in ${srcDir}/* 
  do
    if ! [ -d ${topoDir} ];
    then
      echo "Skipping non-dir file ${topoDir} in ${srcDir}..."
    else
      topoDirTail=${topoDir##*/}
      mkdir -p ${dstDir}/${topoDirTail}
  
      for file in ${topoDir}/* 
        do
          newName=${file##*/}
#          sort -t',' -g -k2,2 -k1,1 -k3,3 ${file} > ${dstDir}/${topoDirTail}/${newName}
          awk -F, 'BEGIN{OFS=","} $2 == 3 \
          { numFired[$3] += $4; \
            numFirings[$3] += $5; \
            fullFirings[$3] += $6;  \
          } \
          END { \
                printf "Module ID, numItems, numFirings, fullFirings, avgOcc, avgOccRatio, avgFullFirings\n"; \
                for (key in numFired) { \
                avgOcc = numFirings[key] == 0 ? 0 : numFired[key]/numFirings[key]; \
                avgOccRatio = avgOcc/128; \
                avgFullFirings = numFirings[key] == 0 ? 0 : fullFirings[key] / numFirings[key]; \
                printf "%d, %d, %d, %d, %.2f, %.2f, %.2f\n", \
                      key, numFired[key], numFirings[key], fullFirings[key], \
                      avgOcc, avgOccRatio, avgFullFirings;
                } \
                printf "Total firings, total fullFirings\n"; \
                for (key in numFired)  {  \
                  totalFirings += numFirings[key];  \
                  totalFullFirings += fullFirings[key];  \
                }  \
                printf "%d, %d\n", \
                  totalFirings, totalFullFirings;
                }'  \
          ${file} | sort > ${dstDir}/${topoDirTail}/${newName}
        done
fi
  done
