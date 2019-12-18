import matplotlib.pyplot as plt
import csv
import numpy

y = []

i = 0

with open('.results/ppo-2019-12-18-01-09-00.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        y.append(float(row[0]))

def movingaverage(interval, window_size):
    window = numpy.ones(int(window_size))/float(window_size)
    return numpy.convolve(interval, window, 'same')

window_size = 1000

y_av = movingaverage(y, window_size)
y_av = y_av[window_size:-window_size]

plt.plot(range(len(y_av)), y_av)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Interesting Graph\nCheck it out')
plt.legend()
plt.show()