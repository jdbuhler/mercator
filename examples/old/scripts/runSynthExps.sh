#!/usr/bin/bash
# Script to run Mercator on synthetic apps 
#  using different parameters.

homeDir=".."

mtr_prog="${homeDir}/bin/MyApp"

# Grab timestamp to be used in output dir name so old files don't
#  get blown away
timestamp=$(date +%s)

outFileDirPrefix="${homeDir}/results_BlackScholes_${timestamp}"
#outFileNamePrefix="out-256K" 
outFileNamePrefix="out" 
############################### 
## Define options per parameter 
## App topology options 
#declare -a topo_arr=("SAMETYPEPIPE" "DIFFTYPEPIPE" "UBERNODEPIPE" "SELFLOOPPIPE")
#declare -a topo_arr=("SAMETYPEPIPE" "DIFFTYPEPIPE" "UBERNODEPIPE" "SELFLOOPPIPE" "REP4SAMEPIPE" "REP4DIFFPIPE" "REP4SEMIDIFFPIPE")
#declare -a topo_arr=("REPSAMEPIPE" "REPDIFFPIPE")
declare -a topo_arr=("DIFFTYPEPIPE" "UBERNODEPIPE")
#declare -a topo_arr=("DIFFTYPEPIPE" "SAMETYPEPIPE")
#declare -a topo_arr=("SELFLOOPPIPE" "SAMETYPEPIPE") 
#declare -a topo_arr=("REPSAMEPIPE" "REPDIFFPIPE")
#declare -a topo_arr=("REP4SAMEPIPE" "REP4DIFFPIPE" "REP4SEMIDIFFPIPE")
#declare -a topo_arr=("REP4SAMEPIPE")

## Elts-to-threads mapping options
#declare -a map_arr=("1TO1")
declare -a map_arr=("1TO4" "4TO1" "1TO1")
#declare -a map_arr=("1TO1" "1TO2" "1TO4" "2TO1" "4TO1")

## Filtering rate options
declare -a rate_arr=("0.0" "0.25" "0.5" "0.75" "1.0")
declare -a rate_nopt_arr=("000" "025" "050" "075" "100")
#declare -a rate_arr=("0.0" "0.5" "0.75")
#declare -a rate_nopt_arr=("000" "050" "075")
#declare -a rate_arr=("0.50" "0.75" "1.00")
#declare -a rate_nopt_arr=("050" "075" "100")
#declare -a rate_arr=("0.25")
#declare -a rate_nopt_arr=("025")

## Num Black-Scholes iterations options
# Testing-sized sets: use when compiling with -g -O0
#declare -a work_arr=("0" "20" "40" "60" "80" "100")
# Bmark-sized sets: use when compiling with -O3 and without -g
#declare -a work_arr=("0" "5000" "10000" "15000" "20000" "25000")
#declare -a work_arr=("100000" "150000" "200000")
#declare -a work_arr=("12500")
declare -a work_arr=("500" "1000" "1500" "2000" "2500" "3000" "3500" "4000"
"4500" "5000")

###############################

## Iterate through topology options
for topo in "${topo_arr[@]}"
  do
    echo "In for loop, topo $topo"

    # First clear all run indicators
    sed -i "1,\$s/\(.*define RUN_.*PIPE\).*/\1 0/" runtime_config.cuh
    sed -i "1,\$s/\(.*define RUN_BLAST.*APP\).*/\1 0/" runtime_config.cuh

    # Now set one for current iteration
    sed -i "1,\$s/\(.*define RUN_$topo\).*/\1 1/" runtime_config.cuh

    # Append app topo option to output file path
    outFileDir=${outFileDirPrefix}/${topo}

    ## Iterate through mapping options
    for map in "${map_arr[@]}"
      do
        echo "In for loop, map $map"

        # First clear all mapping indicators
        sed -i "1,\$s/\(.*define MAPPING_.*\)[01]\$/\10/" runtime_config.cuh

        # Now set one for current iteration
        sed -i "1,\$s/\(.*define MAPPING_$map\).*/\1 1/" runtime_config.cuh

        ## Iterate through filtering rates
        # NB: use indices to be able to access parallel floating-pt
        #     and no-floating-pt arrays
        numRates="${#rate_arr[@]}"
        for((rateIdx=0; rateIdx<$numRates; ++rateIdx));
        do
          rate=${rate_arr[$rateIdx]}
          rateNoPt=${rate_nopt_arr[$rateIdx]}
          echo "In for loop, filtering rate ${rate}"

          # Set rate for current iteration
          sed -i "1,\$s/\(.*define FILTER_RATE\).*/\1 ${rate}/" runtime_config.cuh

          ## Iterate through Black-Scholes work iterations
          for work in "${work_arr[@]}"
          do
            echo "In for loop, BS work loops $work"

            # Set num work iters for current iteration
            # NB: All work iters counts based off ManyThreads case
            sed -i "1,\$s/\(.*define WORK_ITERS_4THREADS\).*/\1 ${work}/" runtime_config.cuh

  
            # Create output dir if necessary
            mkdir -p $outFileDir

            ## Parallel vs. seql gather/scatter
            ##############################
            # Sequential
#            sed -i "1,\$s/\(.*define SEQUENTIAL_GATHER.*\)[01]\$/\11/" runtime_config.cuh
#            sed -i "1,\$s/\(.*define SEQUENTIAL_SCATTER.*\)[01]\$/\11/" runtime_config.cuh
#            outFileName=${outFileNamePrefix}_${map}_filter${rateNoPt}_work${work}_seql.txt
#            # test
#            echo "outfile dir: $outFileDir outfilename: $outFileName"
#  
#            # build and run app
#            pushd ${homeDir} 
#            rm bin/MyApp
#            make clean && make
#            popd
#            echo -n "Running with topology ${topo} mapping ${map} filtering rate ${rate} work iterations ${work} sequential gather/scatter..."
#            ${mtr_prog} >& ${outFileDir}/${outFileName}
#
#            #test
#            #touch ${outFileDir}/${outFileName}
#
#            echo "finished."
            ##############################
  
            ##############################
            # Parallel
            sed -i "1,\$s/\(.*define SEQUENTIAL_GATHER.*\)[01]\$/\10/" runtime_config.cuh
            sed -i "1,\$s/\(.*define SEQUENTIAL_SCATTER.*\)[01]\$/\10/" runtime_config.cuh
            outFileName=${outFileNamePrefix}_${map}_filter${rateNoPt}_work${work}_parallel.txt
            # test
            echo "outfile dir: $outFileDir outfilename: $outFileName"
            # build and run app
            pushd ${homeDir} 
            rm bin/MyApp
            make clean && make
#            make
            popd
            echo -n "Running with topology ${topo} mapping ${map} filtering rate ${rate} work iterations ${work} parallel gather/scatter..."
            ${mtr_prog} >& ${outFileDir}/${outFileName}

            #test
            #touch ${outFileDir}/${outFileName}

            echo "finished."
            ##############################

          done
        done
      done
  done

