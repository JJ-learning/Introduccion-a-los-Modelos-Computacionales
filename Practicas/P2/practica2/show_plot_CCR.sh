#! /bin/bash

cat << _end_ | gnuplot
set terminal postscript eps color
set output 'CCR.eps'
set key right bottom
set yrange [0:200]
set xrange [0:300]
set xlabel "Iteration"
set ylabel "CCR"
set datafile sep ','
plot 'ccr.txt' using 1:2 t "Train" w l, 'ccr.txt' using 1:3 t "Test" w l lw 2
_end_