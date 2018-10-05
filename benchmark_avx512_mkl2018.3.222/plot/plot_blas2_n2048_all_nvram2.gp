set terminal epslatex standalone color ', 8' header '\renewcommand{\encodingdefault}{T1}\renewcommand{\familydefault}{phv}\renewcommand{\seriesdefault}{l}\renewcommand{\shapedefault}{n}'
set output 'blas2_n2048_all_nvram2.tex'

set size nosquare 0.9, 0.95
set multiplot

#set style fill pattern 9 border
set style fill solid border -1
set boxwidth 0.15

set key revers Left at graph 1.23, 0.93 samplen 1.0

set xrange [0.1:6.9]
set xtics ('8'1,'16'2,'32'3,'64'4,'128'5,'256'6) offset 0.0, 0.0 scale 1.5

set yrange [0:300]
set ytics ('0'0,'50'50,'100'100,'150'150,'200'200,'250'250,'300'300) offset 0.0, 0.0 scale 1.5
set ylabel 'GFLOPS'

set size nosquare 0.475, 0.482
set title '\textbf{gemv (n=2048)}' offset 0.0, -0.5
set origin 0.0, 0.5
f(x) = 016.7043
plot f(x) with lines lt -1 title '\small IEEE fp64 (MKL reference)', \
'./blas2_gemv_nvram2.txt' index 7 using (column(0)+0.70):8 with boxes lt -1 lc rgb '#2691cb' title '\small IEEE fp64', \
'./blas2_gemv_nvram2.txt' index 8 using (column(0)+0.85):8 with boxes lt -1 lc rgb '#9fcdff' title '\small IEEE fp32', \
'./blas2_gemv_nvram2.txt' index 9 using (column(0)+1.00):8 with boxes lt -1 lc rgb '#9988aa' title '\small bfloat16', \
'./blas2_gemv_nvram2.txt' index 10 using (column(0)+1.15):8 with boxes lt -1 lc rgb '#699962' title '\small fixed 16 bit', \
'./blas2_gemv_nvram2.txt' index 11 using (column(0)+1.30):8 with boxes lt -1 lc rgb '#8acd67' title '\small fixed 8 bit', \
'./blas2_gemv_nvram2.txt' index 7 using (column(0)+0.70):8:9 with errorbars lt -1 notitle, \
'./blas2_gemv_nvram2.txt' index 8 using (column(0)+0.85):8:9 with errorbars lt -1 notitle, \
'./blas2_gemv_nvram2.txt' index 9 using (column(0)+1.00):8:9 with errorbars lt -1 notitle, \
'./blas2_gemv_nvram2.txt' index 10 using (column(0)+1.15):8:9 with errorbars lt -1 notitle, \
'./blas2_gemv_nvram2.txt' index 11 using (column(0)+1.30):8:9 with errorbars lt -1 notitle, \
f(x) with line lt -1 notitle

set xlabel 'block size $\tilde\textsl{\textit{n}}$'
set size nosquare 0.475, 0.520
set title '\textbf{spmv (n=2048)}' offset 0.0, -0.5
set origin 0.0, 0.0
h(x) = 033.3341
plot './blas2_smv_nvram2.txt' index 7 using (column(0)+0.70):8 with boxes lt -1 lc rgb '#2691cb' notitle, \
'./blas2_smv_nvram2.txt' index 8 using (column(0)+0.85):8 with boxes lt -1 lc rgb '#9fcdff' notitle, \
'./blas2_smv_nvram2.txt' index 9 using (column(0)+1.00):8 with boxes lt -1 lc rgb '#9988aa' notitle, \
'./blas2_smv_nvram2.txt' index 10 using (column(0)+1.15):8 with boxes lt -1 lc rgb '#699962' notitle, \
'./blas2_smv_nvram2.txt' index 11 using (column(0)+1.30):8 with boxes lt -1 lc rgb '#8acd67' notitle, \
'./blas2_smv_nvram2.txt' index 7 using (column(0)+0.70):8:9 with errorbars lt -1 notitle, \
'./blas2_smv_nvram2.txt' index 8 using (column(0)+0.85):8:9 with errorbars lt -1 notitle, \
'./blas2_smv_nvram2.txt' index 9 using (column(0)+1.00):8:9 with errorbars lt -1 notitle, \
'./blas2_smv_nvram2.txt' index 10 using (column(0)+1.15):8:9 with errorbars lt -1 notitle, \
'./blas2_smv_nvram2.txt' index 11 using (column(0)+1.30):8:9 with errorbars lt -1 notitle, \
h(x) with line lt -1 notitle

unset ylabel
set size nosquare 0.450, 0.520
set title '\textbf{tpsv (n=2048)}' offset 0.0, -0.5
set origin 0.47, 0.0
h(x) = 018.9964
plot './blas2_slv_nvram2.txt' index 7 using (column(0)+0.70):8 with boxes lt -1 lc rgb '#2691cb' notitle, \
'./blas2_slv_nvram2.txt' index 8 using (column(0)+0.85):8 with boxes lt -1 lc rgb '#9fcdff' notitle, \
'./blas2_slv_nvram2.txt' index 9 using (column(0)+1.00):8 with boxes lt -1 lc rgb '#9988aa' notitle, \
'./blas2_slv_nvram2.txt' index 10 using (column(0)+1.15):8 with boxes lt -1 lc rgb '#699962' notitle, \
'./blas2_slv_nvram2.txt' index 11 using (column(0)+1.30):8 with boxes lt -1 lc rgb '#8acd67' notitle, \
'./blas2_slv_nvram2.txt' index 7 using (column(0)+0.70):8:9 with errorbars lt -1 notitle, \
'./blas2_slv_nvram2.txt' index 8 using (column(0)+0.85):8:9 with errorbars lt -1 notitle, \
'./blas2_slv_nvram2.txt' index 9 using (column(0)+1.00):8:9 with errorbars lt -1 notitle, \
'./blas2_slv_nvram2.txt' index 10 using (column(0)+1.15):8:9 with errorbars lt -1 notitle, \
'./blas2_slv_nvram2.txt' index 11 using (column(0)+1.30):8:9 with errorbars lt -1 notitle, \
h(x) with line lt -1 notitle

unset xlabel
set size nosquare 0.45, 0.482
set title '\textbf{tpmv (n=2048)}' offset 0.0, -0.5
set origin 0.47, 0.5
g(x) = 018.6414
plot './blas2_tmv_nvram2.txt' index 7 using (column(0)+0.70):8 with boxes lt -1 lc rgb '#2691cb' notitle, \
'./blas2_tmv_nvram2.txt' index 8 using (column(0)+0.85):8 with boxes lt -1 lc rgb '#9fcdff' notitle, \
'./blas2_tmv_nvram2.txt' index 9 using (column(0)+1.00):8 with boxes lt -1 lc rgb '#9988aa' notitle, \
'./blas2_tmv_nvram2.txt' index 10 using (column(0)+1.15):8 with boxes lt -1 lc rgb '#699962' notitle, \
'./blas2_tmv_nvram2.txt' index 11 using (column(0)+1.30):8 with boxes lt -1 lc rgb '#8acd67' notitle, \
'./blas2_tmv_nvram2.txt' index 7 using (column(0)+0.70):8:9 with errorbars lt -1 notitle, \
'./blas2_tmv_nvram2.txt' index 8 using (column(0)+0.85):8:9 with errorbars lt -1 notitle, \
'./blas2_tmv_nvram2.txt' index 9 using (column(0)+1.00):8:9 with errorbars lt -1 notitle, \
'./blas2_tmv_nvram2.txt' index 10 using (column(0)+1.15):8:9 with errorbars lt -1 notitle, \
'./blas2_tmv_nvram2.txt' index 11 using (column(0)+1.30):8:9 with errorbars lt -1 notitle, \
g(x) with line lt -1 notitle

unset multiplot
set output
