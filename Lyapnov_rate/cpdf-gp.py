#set terminal x11d
#rc_err=0.07140703008014705
#rc_lyapnov=-0.656803
#
#
#
#### Plot 1 ##
#set grid
#set tics font "Arial,10"
#set title font "Arial,13"
#set xlabel font "Arial,13"
#set ylabel font "Arial,13"
#set y2label font "Arial,13"
#set zlabel font "Arial,13"
##set ylabel "y"
#set xlabel "lyapnov"
#set title "cpdf"
#set ylabel "pdf"
#set style fill solid
#
#
##set logscal y
#set yrange[0:0.13]
#set xrange[-1.0:0.2]
##plot "./data_out/lyapnov_end.dat" using 2:4 with p  pt 3 ps 1 lc 15 title "node-010"
#
#plot "./c-pdf_renban/conditiona_pdf.1500" using 1:2 with boxes notitle  lc rgb "red"
##plot "./data_renban/lyap-err_epoch.0001" using 4:3 with p  pt 7 ps 0.4 lc 15 title "err-lyapnov(epoch)"
#

#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################
set terminal gif animate delay 5 optimize size 640,640
set output './data_image/cpdf.gif'
set grid
set tics font "Arial,10"
set title font "Arial,13"
set xlabel font "Arial,13"
set ylabel font "Arial,13"
set y2label font "Arial,13"
set zlabel font "Arial,13"
#set ylabel "y"
set xlabel "lyapnov"
set title "cpdf"
set ylabel "pdf"
set style fill solid


#set logscal y
set yrange[0:0.13]
set xrange[-1:0.2]

set key below
do for [i=1:1500:1] {

filename = sprintf("./c-pdf_renban/conditiona_pdf.%04d", i) # n番目のデータファイルの名前の生成

time = sprintf("t=%d[epoch]", i)
set title time
plot filename using 1:2 with boxes notitle  lc rgb "red"
#plot filename using 4:3 with p  pt 7 ps 0.4 lc 2  title "err-lyapnov(ini)",\
#     filename using 1:3 with p  pt 7 ps 0.4 lc 15 title "err-lyapnov(epoch)"

#
}
unset output