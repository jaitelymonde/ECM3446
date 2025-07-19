# Send output to a PNG file
set terminal png enhanced 
# Set the name of the output file
set output "average.png"

# Set ranges and labels for axes
set xrange [0:30.0]
set yrange [0:*]
set xlabel "x (m)"
set ylabel "averaged u"

# Enforce an aspect ratio of 1
set size square

# Plot the data
plot "average.dat" with lines linecolor rgb "blue"

# End of file
