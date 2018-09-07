#!/usr/bin/bash
# Script to run Mercator on Blast apps
#  using different parameters.

homeDir=".."

mtr_prog="./bin/MyApp"

# Grab timestamp to be used in output dir name so old files don't
#  get blown away
timestamp=$(date +%s)

outFileDirPrefix="results_Blast_${timestamp}"
outFileNamePrefix="out" 

############################### 
## Define options per parameter 
## App topology options 
#declare -a topo_arr=("BLASTAPP" "BLASTUBERAPP") 
declare -a topo_arr=("BLASTAPP") 
#declare -a topo_arr=("BLASTAPP" "BLAST2MODULESAPP") 

## Elts-to-threads mapping options
#declare -a map_arr=("1TO1" "2TO1" "4TO1")
declare -a map_arr=("1TO1")

## Size of input query set, in Kbases
#declare -a size_arr=("2" "4" "6" "8" "10" "20" "30" "40" "50")
declare -a size_arr=("2" "4" "6" "8" "10")
#declare -a size_arr=("20")

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

    ## Iterate through query size options
    for size in "${size_arr[@]}"
      do
        echo "In for loop, size $size"

        # Set query size for current iteration
        sed -i "1,\$s/\(\#define QUERYSIZE_K\).*/\1 $size/" runtime_config.cuh

        ## Iterate through mapping options
        for map in "${map_arr[@]}"
          do
            echo "In for loop, map $map"

            # First clear all mapping indicators
            sed -i "1,\$s/\(.*define MAPPING_.*\)[01]\$/\10/" runtime_config.cuh

            # Now set one for current iteration
            sed -i "1,\$s/\(.*define MAPPING_$map\).*/\1 1/" runtime_config.cuh

                # Create output dir if necessary

                ## Parallel vs. seql gather/scatter
                ##############################
                # Sequential
#                sed -i "1,\$s/\(.*define SEQUENTIAL_GATHER.*\)[01]\$/\11/" runtime_config.cuh
#                sed -i "1,\$s/\(.*define SEQUENTIAL_SCATTER.*\)[01]\$/\11/" runtime_config.cuh
#                outFileName=${outFileNamePrefix}_${size}k_${map}_filter${rateNoPt}_work${work}_seql.txt
#                # test
#                echo "outfile dir: $outFileDir outfilename: $outFileName"
#  
#                # build and run app
#                pushd ${homeDir} 
#                rm bin/MyApp
#                make clean && make
#                echo -n "Running with topology ${topo} mapping ${map} filtering rate ${rate} work iterations ${work} sequential gather/scatter..."
#                ${mtr_prog} >& ${outFileDir}/${outFileName}
#                popd
#
#                #test
#                #touch ${outFileDir}/${outFileName}
#
#                echo "finished."
                ##############################
  
                ##############################
                # Parallel
                sed -i "1,\$s/\(.*define SEQUENTIAL_GATHER.*\)[01]\$/\10/" runtime_config.cuh
                sed -i "1,\$s/\(.*define SEQUENTIAL_SCATTER.*\)[01]\$/\10/" runtime_config.cuh
                outFileName=${outFileNamePrefix}_${size}k_${map}_parallel.txt
                # test
                echo "outfile dir: $outFileDir outfilename: $outFileName"
                # build and run app
                pushd ${homeDir} 
                mkdir -p $outFileDir
                rm bin/MyApp
                make clean && make
                echo -n "Running with topology ${topo} mapping ${map} parallel gather/scatter..."
                ${mtr_prog} >& ${outFileDir}/${outFileName}
                popd

                #test
#                touch ${outFileDir}/${outFileName}

                echo "finished."
                ##############################
          done
      done
  done

