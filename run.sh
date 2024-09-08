#!bin/bash/

g++ -O3 src/main.cpp -o src/main.o
src/main.o --n 100 --m 10 --k 10 --p 0.7 --q 0.5 --r 0.5 --a 30
rm src/main.o