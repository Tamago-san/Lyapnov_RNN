#! /bin/sh
#$ -cwd
#$ -V -S /bin/bash
#$ -pe smp 20
#$ -N RNN_err-ly
mpirun -np 20 ./a.out
