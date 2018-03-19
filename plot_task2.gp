# example: plotting the result of task 2 by using gnuplot
set term  png
set output "task2_result.png"
set view map
set size ratio -1
splot "Output_task2.txt" w p pt 5 pal
exit
