from skfeature.function.similarity_based.fisher_score import fisher_score
from scipy.fftpack import fft
import numpy as np
import matplotlib.pyplot as plt



#input_file = open('/mnt/c/Users/Mortagetti/Desktop/Sample_gyroscope_t1.txt')
input_file = open('Sample_gyroscope_t1.txt')
for i in range(0,8):
    print(input_file.readline())

time =  []
x    =  []
y    =  []
z    =  []

for line in input_file:
#    print(line)
    if len(line) > 0:

        parts = line.split()

        time.append(parts[0])
        x.append(parts[1])
        y.append(parts[2])
        z.append(parts[3])

time.pop(0)
x.pop(0)
y.pop(0)
z.pop(0)

x = np.array(x)
y = np.array(y)
z = np.array(z)

# Number of sample points
N = 258

# Sample spacing
T = 1.0 / 800.0



acceleration_data = np.concatenate((x,y,z), axis=0)

print(acceleration_data)
#fisher_score()
ft = fft(acceleration_data)

plt.plot(time[1:N], 2.0/N * np.abs(ft[1:N]), '-r')
plt.grid()
plt.show()