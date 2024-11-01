#!/bin/bash

# Define the range of iterations
start_iterate=0 # for example, starting from 100
end_iterate=20000 # for example, ending at 30100

# Define the number of threads
THREADS=5

# Calculate the total iterations and iterations per thread
TOTAL_ITERATIONS=$((end_iterate - start_iterate))
ITERATIONS_PER_THREAD=$((TOTAL_ITERATIONS / THREADS))

# Run the Python script in parallel
for i in $(seq 0 $((THREADS - 1)))
do
   start=$((start_iterate + i * ITERATIONS_PER_THREAD))
   end=$((start + ITERATIONS_PER_THREAD))

   # For the last thread, adjust to ensure it covers up to end_iterate
   if [ $i -eq $((THREADS - 1)) ]; then
       end=$end_iterate
   fi

   python data_generation.py --start $start --end $end &
done

wait # Wait for all background processes to finish
