#!bin/bash/

g++ -O3 src/main.cpp -o src/main.o -fopenmp
# time src/main.o --n 100 --m 10 --k 10 --p 0.7 --q 0.3 --r 0.5 --a 30 --threads 1
# time src/main.o --n 100 --m 10 --k 10 --p 0.75 --q 0.25 --r 0.5 --a 33 --b 10 --threads 1
# time src/main.o --n 100 --m 0.1 --k 0.1 --p 0.75 --q 0.25 --r 0.5 --a 0.5 --b 0.05 --d 0.1 --threads 1 --mode plot
# time src/main.o --n 100 --m 0.1 --k 0.1 --p 0.75 --q 0.25 --r 0.5 --a 0.3 --b 0.05 --threads 1 --mode regress
time src/main.o --p 0.75 --q 0.25 --r 0.3 --d 0.1 --threads 20 --mode regress
# time src/main.o --p 0.75 --q 0.25 --r 0.7 --threads 18 --mode regress
rm src/main.o