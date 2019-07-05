#set terminal x11d
#rc_err=0.07140703008014705
#rc_lyapnov=-0.656803



## Plot 1 ##
set grid
set tics font "Arial,10"
set title font "Arial,15"
set xlabel font "Arial,15"
set ylabel font "Arial,15"
set y2label font "Arial,15"
set zlabel font "Arial,15"
#set ylabel "y"
set xlabel "LYAPNOV"
set title "node50-epoch1500"
set ylabel "ERROR"
#set logscal y
#set yrange[0.01:0.1]
set logscal y
set nokey
set yrange[0.01:1.0]
set xrange[-10.0:0.1]


plot "./data_renban/lyap-err_epoch.1500" using 1:3 with p  pt 7 ps 0.5 lc 15 title "node-050"