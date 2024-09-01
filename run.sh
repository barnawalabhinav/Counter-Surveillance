#!bin/bash/

g++ -O3 src/main.cpp -o src/main.o
src/main.o --n 100 --m 10 --k 5 --p 1.0 --q 0.0
rm src/main.o