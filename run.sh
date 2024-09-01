#!bin/bash/

g++ -O3 src/main.cpp -o src/main.o
src/main.o --n 5
rm src/main.o