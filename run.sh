#!bin/bash/

g++ -O3 src/main.cpp -o src/main.o -fopenmp
# time src/main.o --n 100 --m 10 --k 10 --p 0.7 --q 0.3 --r 0.5 --a 30 --threads 1
time src/main.o --n 100 --m 10 --k 10 --p 0.75 --q 0.25 --r 0.5 --a 33 --b 10 --threads 1
rm src/main.o