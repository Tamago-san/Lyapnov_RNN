set terminal gif animate delay 5 optimize size 640,640
set output './Lyapnov_rate/data_image/err-lyap_epo1500_11.gif'
set tics font "Arial,10"
set title font "Arial,15"
set xlabel font "Arial,15"
set ylabel font "Arial,15"
set y2label font "Arial,15"
set zlabel font "Arial,15"
set xlabel "LYAPNOV"
set title "ERROR-LYAPNOV"
set ylabel "ERROR"
#set yrange[-3:4]
#set logscal y
#set yrange[0.01:1]
#set xrange[-1.0:0.1]
set xrange[-0.2:0.1]
set yrange[0.01:0.1]
#set yrange[0.01:0.2]
#set xrange[-0.45:0.05]

set key below
do for [i=1:1500:1] {

filename = sprintf("./data_renban/lyap-err_epoch.%04d", i) # n番目のデータファイルの名前の生成

time = sprintf("t=%d[epoch]", i)
set title time
plot filename using 1:3 with p  pt 7 ps 0.5 lc 15 title "node-050"
#plot filename using 4:3 with p  pt 7 ps 0.4 lc 2  title "err-lyapnov(ini)",\
#     filename using 1:3 with p  pt 7 ps 0.4 lc 15 title "err-lyapnov(epoch)"

#
}
unset output