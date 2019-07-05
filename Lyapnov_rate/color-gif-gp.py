set terminal gif animate delay 5 optimize size 640,640

set output './data_image/err-lyap_epo1500_1.gif'
set object 1 rect behind from screen 0,0 to screen 1,1 fc rgb "#333631" fillstyle solid 1.0

set pm3d
set pm3d map
set ticslevel 0
set size square
set tics font "Arial,11"
set title font "Arial,11"
set xlabel tc rgb "white" font ",12"
set ylabel tc rgb "white" font ",12" offset -0.3, 0
set y2label font "Arial,11"
set zlabel font "Arial,11"
set title  textcolor rgb "white"
set xlabel "LYAPNOV"
#set title "ERROR-LYAPNOV"
set ylabel "ERROR"

#set logscale cb
#set cbrange[0:0.0001]
set xrange[-0.2:0.1]
set yrange[0.01:0.1]
set cbrange[0.9:70]
#set pm3d interpolate 4,4
set logscale cb

set border lc rgb "white"
set grid lc rgb "gray" lt 2
set palette maxcolors 5
#set palette rgbformulae 21,22,23
set palette defined (0 "black",1 "blue",2 "red",3 "yellow",4 "green")


#set key below
do for [i=1:1500] {

filename = sprintf("./pdf_renban/probability_density_function.%04d", i) # n番目のデータファイルの名前の生成

time = sprintf("t=%d[epoch]", i)
set title time
set palette maxcolors 5
splot filename u 1:2:3 with pm3d
#plot filename using 4:3 with p  pt 7 ps 0.4 lc 15 title "err-lyapnov(epoch)"
#plot filename using 4:3 with p  pt 7 ps 0.4 lc 2  title "err-lyapnov(ini)",\
#     filename using 1:3 with p  pt 7 ps 0.4 lc 15 title "err-lyapnov(epoch)"

#
}
unset output