set object 1 rect behind from screen 0,0 to screen 1,1 fc rgb "#333631" fillstyle solid 1.0

set pm3d
set pm3d map
set ticslevel 0
set size square
set tics font "Arial,11"
set title font "Arial,11"
set xlabel tc rgb "white" font ",12"
set ylabel tc rgb "white" font ",12" offset -0.3, 0
set title  textcolor rgb "white"
set y2label font "Arial,11"
set zlabel font "Arial,11"
set xlabel "LYAPNOV"
#set title "ERROR-LYAPNOV"
set ylabel "ERROR"


#set logscale cb
#set cbrange[0:0.0001]
set xrange[-0.2:0.1]
set yrange[0.01:0.1]
set cbrange[0.7:70]
#set pm3d interpolate 4,4
set logscale cb

set border lc rgb "white"
set grid lc rgb "gray" lt 2


#set rmargin 10
#set colorbox vertical user origin .02,.1 size .8,.04

#set cbtics 1
#set cblabel "Temp" tc rgb "black" font ",30"
set palette maxcolors 5
#set palette rgbformulae 21,22,23
set palette defined (0 "black",1 "blue",2 "red",3 "yellow",4 "green")
splot "./pdf_renban/probability_density_function.0350" u 1:2:3 with pm3d notitle


##reset
##set size square            # same side lengths for x and y
##set xlabel 'lyapnov'             # x-axis
##set ylabel 'error'             # y-axis
###set cbrange[0:0.001]
#set xrange[-0.15:0.]
#set yrange[0.01:0.1]
#set nosurface              # do not show surface plot
#unset ztics                # do not show z-tics
#set contour base           # enables contour lines
##set cntrparam levels 4    # draw 10 contours
#set cntrparam levels discrete 2,7,15
#set view 0,0               # view from the due north
#splot "./data_renban2/probability_density_function.1400" u 1:2:3 with l lw 3
#pause -1
#