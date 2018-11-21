set terminal epslatex standalone color ', 8' header '\renewcommand{\encodingdefault}{T1}\renewcommand{\familydefault}{phv}\renewcommand{\seriesdefault}{l}\renewcommand{\shapedefault}{n}'
set output 'blas2_n447_avx512.tex'

set size nosquare 1.37, 1.01
set multiplot

set style fill solid border -1
set boxwidth 0.15

set key revers Left at graph 2.75, 1.0 samplen 0.8
set bars 0.25

set xrange [0.25:6.75]
set xtics ('8'1,'16'2,'32'3,'64'4,'128'5,'256'6) offset 0.0, 0.0 scale 1.5

set yrange [0:300]
set ytics ('0'0,'50'50,'100'100,'150'150,'200'200,'250'250,'300'300) offset 0.0, 0.0 scale 1.5 nomirror
set ylabel 'GFLOPS' offset 0.5, 0.0

set y2range [0:18]
set y2tics ('0'8,'2'10,'4'12,'6'14,'8'16) nomirror
set y2label 'gain' offset -1.5, 2.0

f(x)=9.0

set size nosquare 0.55, 0.50
set title '\textbf{gemv (n=447)}' offset 0.0, -0.5
set origin 0.0, 0.53
plot './blas2_gemv_avx512.txt' index 0 using (column(0)+0.70):8 with boxes lt -1 lc rgb '#2691cb' notitle, \
'./blas2_gemv_avx512.txt' index 1 using (column(0)+0.85):8 with boxes lt -1 lc rgb '#9fcdff' notitle, \
'./blas2_gemv_avx512.txt' index 2 using (column(0)+1.00):8 with boxes lt -1 lc rgb '#9988aa' notitle, \
'./blas2_gemv_avx512.txt' index 3 using (column(0)+1.15):8 with boxes lt -1 lc rgb '#699962' notitle, \
'./blas2_gemv_avx512.txt' index 4 using (column(0)+1.30):8 with boxes lt -1 lc rgb '#8acd67' notitle, \
'./blas2_gemv_avx512.txt' index 0 using (column(0)+0.70):8:9 with errorbars lt -1 notitle, \
'./blas2_gemv_avx512.txt' index 1 using (column(0)+0.85):8:9 with errorbars lt -1 notitle, \
'./blas2_gemv_avx512.txt' index 2 using (column(0)+1.00):8:9 with errorbars lt -1 notitle, \
'./blas2_gemv_avx512.txt' index 3 using (column(0)+1.15):8:9 with errorbars lt -1 notitle, \
'./blas2_gemv_avx512.txt' index 4 using (column(0)+1.30):8:9 with errorbars lt -1 notitle, \
'./blas2_gemv_gains_avx512.txt' index 0 using (0.70+0.15*column(0)):(8+$3):4 with errorlines linestyle 7 lw 2 lc 7 ps 0.7 axes x1y2 notitle, \
'./blas2_gemv_gains_avx512.txt' index 0 using (1.70+0.15*column(0)):(8+$5):6 with errorlines linestyle 7 lw 2 lc 7 ps 0.7 axes x1y2 notitle, \
'./blas2_gemv_gains_avx512.txt' index 0 using (2.70+0.15*column(0)):(8+$7):8 with errorlines linestyle 7 lw 2 lc 7 ps 0.7 axes x1y2 notitle, \
'./blas2_gemv_gains_avx512.txt' index 0 using (3.70+0.15*column(0)):(8+$9):10 with errorlines linestyle 7 lw 2 lc 7 ps 0.7 axes x1y2 notitle, \
'./blas2_gemv_gains_avx512.txt' index 0 using (4.70+0.15*column(0)):(8+$11):12 with errorlines linestyle 7 lw 2 lc 7 ps 0.7 axes x1y2 notitle, \
'./blas2_gemv_gains_avx512.txt' index 0 using (5.70+0.15*column(0)):(8+$13):14 with errorlines linestyle 7 lw 2 lc 7 ps 0.7 axes x1y2 notitle, \
f(x) with lines linestyle 7 axes x1y2 notitle

unset ylabel
unset y2label

set xlabel 'block size $\tilde\textsl{\textit{n}}$'
set size nosquare 0.505, 0.538
set title '\textbf{spmv (n=447)}' offset 0.0, -0.5
set origin 0.022, 0.0
plot './blas2_spmv_avx512.txt' index 0 using (column(0)+0.70):8 with boxes lt -1 lc rgb '#2691cb' notitle, \
'./blas2_spmv_avx512.txt' index 1 using (column(0)+0.85):8 with boxes lt -1 lc rgb '#9fcdff' notitle, \
'./blas2_spmv_avx512.txt' index 2 using (column(0)+1.00):8 with boxes lt -1 lc rgb '#9988aa' notitle, \
'./blas2_spmv_avx512.txt' index 3 using (column(0)+1.15):8 with boxes lt -1 lc rgb '#699962' notitle, \
'./blas2_spmv_avx512.txt' index 4 using (column(0)+1.30):8 with boxes lt -1 lc rgb '#8acd67' notitle, \
'./blas2_spmv_avx512.txt' index 0 using (column(0)+0.70):8:9 with errorbars lt -1 notitle, \
'./blas2_spmv_avx512.txt' index 1 using (column(0)+0.85):8:9 with errorbars lt -1 notitle, \
'./blas2_spmv_avx512.txt' index 2 using (column(0)+1.00):8:9 with errorbars lt -1 notitle, \
'./blas2_spmv_avx512.txt' index 3 using (column(0)+1.15):8:9 with errorbars lt -1 notitle, \
'./blas2_spmv_avx512.txt' index 4 using (column(0)+1.30):8:9 with errorbars lt -1 notitle, \
'./blas2_spmv_gains_avx512.txt' index 0 using (0.70+0.15*column(0)):(8+$3):4 with errorlines linestyle 7 lw 2 lc 7 ps 0.7 axes x1y2 notitle, \
'./blas2_spmv_gains_avx512.txt' index 0 using (1.70+0.15*column(0)):(8+$5):6 with errorlines linestyle 7 lw 2 lc 7 ps 0.7 axes x1y2 notitle, \
'./blas2_spmv_gains_avx512.txt' index 0 using (2.70+0.15*column(0)):(8+$7):8 with errorlines linestyle 7 lw 2 lc 7 ps 0.7 axes x1y2 notitle, \
'./blas2_spmv_gains_avx512.txt' index 0 using (3.70+0.15*column(0)):(8+$9):10 with errorlines linestyle 7 lw 2 lc 7 ps 0.7 axes x1y2 notitle, \
'./blas2_spmv_gains_avx512.txt' index 0 using (4.70+0.15*column(0)):(8+$11):12 with errorlines linestyle 7 lw 2 lc 7 ps 0.7 axes x1y2 notitle, \
'./blas2_spmv_gains_avx512.txt' index 0 using (5.70+0.15*column(0)):(8+$13):14 with errorlines linestyle 7 lw 2 lc 7 ps 0.7 axes x1y2 notitle, \
f(x) with lines linestyle 7 axes x1y2 notitle

unset ylabel
set size nosquare 0.505, 0.538
set title '\textbf{tpsv (n=447)}' offset 0.0, -0.5
set origin 0.55, 0.0
plot './blas2_tpsv_avx512.txt' index 0 using (column(0)+0.70):8 with boxes lt -1 lc rgb '#2691cb' notitle, \
'./blas2_tpsv_avx512.txt' index 1 using (column(0)+0.85):8 with boxes lt -1 lc rgb '#9fcdff' notitle, \
'./blas2_tpsv_avx512.txt' index 2 using (column(0)+1.00):8 with boxes lt -1 lc rgb '#9988aa' notitle, \
'./blas2_tpsv_avx512.txt' index 3 using (column(0)+1.15):8 with boxes lt -1 lc rgb '#699962' notitle, \
'./blas2_tpsv_avx512.txt' index 4 using (column(0)+1.30):8 with boxes lt -1 lc rgb '#8acd67' notitle, \
'./blas2_tpsv_avx512.txt' index 0 using (column(0)+0.70):8:9 with errorbars lt -1 notitle, \
'./blas2_tpsv_avx512.txt' index 1 using (column(0)+0.85):8:9 with errorbars lt -1 notitle, \
'./blas2_tpsv_avx512.txt' index 2 using (column(0)+1.00):8:9 with errorbars lt -1 notitle, \
'./blas2_tpsv_avx512.txt' index 3 using (column(0)+1.15):8:9 with errorbars lt -1 notitle, \
'./blas2_tpsv_avx512.txt' index 4 using (column(0)+1.30):8:9 with errorbars lt -1 notitle, \
'./blas2_tpsv_gains_avx512.txt' index 0 using (0.70+0.15*column(0)):(8+$3):4 with errorlines linestyle 7 lw 2 lc 7 ps 0.7 axes x1y2 notitle, \
'./blas2_tpsv_gains_avx512.txt' index 0 using (1.70+0.15*column(0)):(8+$5):6 with errorlines linestyle 7 lw 2 lc 7 ps 0.7 axes x1y2 notitle, \
'./blas2_tpsv_gains_avx512.txt' index 0 using (2.70+0.15*column(0)):(8+$7):8 with errorlines linestyle 7 lw 2 lc 7 ps 0.7 axes x1y2 notitle, \
'./blas2_tpsv_gains_avx512.txt' index 0 using (3.70+0.15*column(0)):(8+$9):10 with errorlines linestyle 7 lw 2 lc 7 ps 0.7 axes x1y2 notitle, \
'./blas2_tpsv_gains_avx512.txt' index 0 using (4.70+0.15*column(0)):(8+$11):12 with errorlines linestyle 7 lw 2 lc 7 ps 0.7 axes x1y2 notitle, \
'./blas2_tpsv_gains_avx512.txt' index 0 using (5.70+0.15*column(0)):(8+$13):14 with errorlines linestyle 7 lw 2 lc 7 ps 0.7 axes x1y2 notitle, \
f(x) with lines linestyle 7 axes x1y2 notitle

unset xlabel
set size nosquare 0.505, 0.5
set title '\textbf{tpmv (n=447)}' offset 0.0, -0.5
set origin 0.55, 0.53
plot './blas2_tpmv_avx512.txt' index 0 using (column(0)+0.70):8 with boxes lt -1 lc rgb '#2691cb' title '\small $\textsl{FP}_\textsl{\tiny 11,52}$', \
'./blas2_tpmv_avx512.txt' index 1 using (column(0)+0.85):8 with boxes lt -1 lc rgb '#9fcdff' title '\small $\textsl{FP}_\textsl{\tiny 8,23}$', \
'./blas2_tpmv_avx512.txt' index 2 using (column(0)+1.00):8 with boxes lt -1 lc rgb '#9988aa' title '\small $\textsl{FP}_\textsl{\tiny 8,7}$ (bfloat16)', \
'./blas2_tpmv_avx512.txt' index 3 using (column(0)+1.15):8 with boxes lt -1 lc rgb '#699962' title '\small fixed point 16 bit', \
'./blas2_tpmv_avx512.txt' index 4 using (column(0)+1.30):8 with boxes lt -1 lc rgb '#8acd67' title '\small fixed point 8 bit', \
'./blas2_tpmv_avx512.txt' index 0 using (column(0)+0.70):8:9 with errorbars lt -1 notitle, \
'./blas2_tpmv_avx512.txt' index 1 using (column(0)+0.85):8:9 with errorbars lt -1 notitle, \
'./blas2_tpmv_avx512.txt' index 2 using (column(0)+1.00):8:9 with errorbars lt -1 notitle, \
'./blas2_tpmv_avx512.txt' index 3 using (column(0)+1.15):8:9 with errorbars lt -1 notitle, \
'./blas2_tpmv_avx512.txt' index 4 using (column(0)+1.30):8:9 with errorbars lt -1 notitle, \
'./blas2_tpmv_gains_avx512.txt' index 0 using (0.70+0.15*column(0)):(8+$3):4 with errorlines linestyle 7 lw 2 lc 7 ps 0.7 axes x1y2 title '\small gain over $\textsl{FP}_\textsl{\tiny 11,52}$ (MKL reference)', \
'./blas2_tpmv_gains_avx512.txt' index 0 using (1.70+0.15*column(0)):(8+$5):6 with errorlines linestyle 7 lw 2 lc 7 ps 0.7 axes x1y2 notitle, \
'./blas2_tpmv_gains_avx512.txt' index 0 using (2.70+0.15*column(0)):(8+$7):8 with errorlines linestyle 7 lw 2 lc 7 ps 0.7 axes x1y2 notitle, \
'./blas2_tpmv_gains_avx512.txt' index 0 using (3.70+0.15*column(0)):(8+$9):10 with errorlines linestyle 7 lw 2 lc 7 ps 0.7 axes x1y2 notitle, \
'./blas2_tpmv_gains_avx512.txt' index 0 using (4.70+0.15*column(0)):(8+$11):12 with errorlines linestyle 7 lw 2 lc 7 ps 0.7 axes x1y2 notitle, \
'./blas2_tpmv_gains_avx512.txt' index 0 using (5.70+0.15*column(0)):(8+$13):14 with errorlines linestyle 7 lw 2 lc 7 ps 0.7 axes x1y2 notitle, \
f(x) with lines linestyle 7 axes x1y2 notitle

unset multiplot
set output
