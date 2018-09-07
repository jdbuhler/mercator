#!/bin/bash 

# AWK script to analyze the timing instrumentation output of MERCATOR
#   with the purpose of gathering the info necessary for the mean-val
#   flow model of performance..
# Expected format of block-wide timing rows in CSV file:
# blockID, instrumentation ID (value currently 0 for block-wide timing), 
#  mainLoopTime (cycles), mainLoopTime (ms), 
#  firingQueueTime (cycles), firingQueueTime (ms)
#
# Expected format of module-wise per-module timing rows in CSV file:
# blockID, instrumentation ID (value currently 0 for block-wide timing), 
# .., fireQueue (cycles), fireQueue (ms)   <-- field 23 (1-based)
#

# Goals: 1) Model building: when prog has been set up to account time from
#            "full" firings only, extract for each module its throughput
#            mu_i.  This becomes the module's stationary throughput in the
#            model. Also extract the fraction of total kernel time spent
#            executing modules as 1-f_0; this becomes the value that all module
#            execution fractions (f_i's) must sum to.  Plug the mu_i's 
#            and 1-f_0 into the model, and using a simultaneous equation solver,
#            solve for the f_i's.  The f_i's give the ideal fraction of time
#            spent executing each module.
#        2) Model validation: when prog has been set up to account time from all
#            firings, extract for each module its empirical f_i value; i.e.,
#            fraction of total execution time spent executing that module.
#            Compare these to the f_i's derived from the model.

MAX_FIRING_WIDTH=128

usage="Usage: $0 <timing results directory>    NB: Results assumed to be generated
by MERCATOR with INSTRUMENT_TIME and INSTRUMENT_TIME_HOST both set to 1."

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
dstDir=${srcDir}_agg_timing

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
# input fields for kernel-level timing: timerType ($2): 0, mainLoopTime ($4), firingQueue ($6)
# input fields for module-level timing: timerType ($2): 1, moduleID ($3),
#                                 firingTime ($23)
# output: Module ID, gatherTime, scatterTime, gather+scatterTime, time in this module, f_i (fraction of time in this
#           module), 1 - f_0 (frac of time executing any module)
#         NB: (1 - f_0) should be same for all modules.
#  coles: Modify output so it gets stats that change with FULLFIRINGS setting!
#  --i.e., grab gather/exec and scatter time separately and sum them together
#  as firing time, rather than getting it directly from higher level
#  Fields 11 & 14
          awk -F, 'BEGIN{OFS=","} \
          $2 == 0 \
          { totalKernelTime += $4; \
            totalFiringQueueTime += $6; \
            if (totalKernelTime > 0) { \
              blockCount += 1;
            } \
          } \
          $2 == 1 \
          { firingTime[$3] += $23; \
            gatherExec[$3] += $11; \
            scatter[$3]    += $14; \
          } \
          END { totalFiringRatio = totalFiringQueueTime/totalKernelTime; \
                avgTotalKernelTime = totalKernelTime/blockCount; \
                avgTotalFiringQueueTime = totalFiringQueueTime/blockCount; \
                printf "Module ID, Avg gather/exec time, avg scatter time, both, totalFiringTime in queue, moduleTimeFrac, totalTimeFrac, totalTime\n"; \
                for (key in firingTime) { \
                  avgGatherExec = gatherExec[key]/blockCount; \
                  avgScatter = scatter[key]/blockCount; \
                  avgBoth = (gatherExec[key] + scatter[key]) / blockCount; \
                  avgFiringTime = firingTime[key]/blockCount; \
                  mu = avgFiringTime/avgTotalKernelTime; \
                  printf "%d, %.2f, %.2f, %.2f, %.2f, %.3f, %.3f, %.3f\n", \
                      key, avgGatherExec, avgScatter, avgBoth, avgFiringTime, mu, totalFiringRatio, avgTotalKernelTime;
                } \
              }'  \
          ${file} | sort -g > ${dstDir}/${topoDirTail}/${newName}
        done
fi
  done
