#! /bin/bash

cat << _end_ | gnuplot
set terminal postscript eps color
set output 'Convergencia.eps'
set key right bottom
set xlabel "Iteration"
set ylabel "Error"
set datafile sep ','
plot 'errores.txt' using 1:2 t "Train" w l, 'errores.txt' using 1:3 t "Test" w l lw 2
_end_