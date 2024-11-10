#!/bin/sh
### Set the job name (for your reference)
#PBS -N project_experiment
### Set the project name, your department code by default
#PBS -P col870.course
### Request email when job begins and ends, don't change anything on the below line 
#PBS -m bea
### Specify email address to use for notification, don't change anything on the below line
#PBS -M $USER@iitd.ac.in
#### Request your resources, just change the numbers
#PBS -l select=1:ncpus=40:mem=64G
### Specify "wallclock time" required for this job, hhh:mm:ss
#PBS -l walltime=48:00:00
#PBS -l software=C++

# After job starts, must goto working directory. 
# $PBS_O_WORKDIR is the directory from where the job is fired. 
echo "==============================="
echo $PBS_JOBID
cat $PBS_NODEFILE
echo "==============================="
cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR

module load compiler/gcc/9.1.0

g++ -O3 src/main.cpp -o src/main_4.o -fopenmp
time src/main_4.o --p 0.75 --q 0.25 --r 0.3 --d 0.9 --threads 40 --mode regress
time src/main_4.o --p 0.75 --q 0.25 --r 0.7 --d 0.9 --threads 40 --mode regress
# time src/main_4.o --p 0.7 --q 0.3 --r 0.3 --threads 1 --mode plot
rm src/main_4.o