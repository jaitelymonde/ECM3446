# This gnuplot script plots the results from the first coursework assignment. 
# It is assumed that the data to be plotted are in a file called
# intial.dat which contains 3 columns: x,y,u
# The plot is sent to a PNG file called intial.png 
# To use this file copy it to the directory/folder containing 
# final.dat and run the command: 
# gnuplot plot_initial 


# Send output to a PNG file
set terminal png  enhanced 
# Set the name of the output file
set output "initial.png"

# Set ranges and labels for axes
set xrange [0:30.0]
set yrange [0:30.0]
set xlabel "x (m)"
set ylabel "y (m)"

# Enforce an aspect ratio of 1
set size square

# Set the range of the colour scale
set cbrange [0:1]

# Plot the data 
plot "initial.dat" with image 

# End of file