from skfeature.function.similarity_based.fisher_score import fisher_score
from scipy.fftpack import fft, rfft, irfft
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import normalize
import sklearn
from pandas import Series
from pandas import DataFrame
from pandas import concat

# Fourier implementation from https://docs.scipy.org/doc/scipy/reference/tutorial/fftpack.html

#input_file = open('/mnt/c/Users/Mortagetti/Desktop/Sample_gyroscope_t1.txt')
input_file = open(r'C:\Users\Mortagetti\Desktop\Notes for School\Summer_Research\gear7_2\MPU6500 Acceleration Sensor.csv', "r")

time = []
x    = []
y    = []
z    = []

for line in input_file:
    if len(line) > 0:

        parts = line.split(',')

        time.append(parts[0])
        x.append(parts[1])
        y.append(parts[2])
        z.append(parts[3])

x = np.array(x)
y = np.array(y)
z = np.array(z)

# Number of sample points for training
N = 1000

# The sample points for testing
T = 250
acceleration_data = np.concatenate((x,y,z), axis=0)

#Normalizing the acceleration data
#acceleration_data = acceleration_data.reshape(-1,1)
#normalized_data = normalize(acceleration_data).ravel()

#fisher_score()
# FT without smoothing
ft = fft(acceleration_data)



# FT with smoothing
rft = rfft(acceleration_data)
y_smooth = irfft(rft)


# Windowing through pandas found on https://machinelearningmastery.com/basic-feature-engineering-time-series-data-python/
series  = Series(np.abs(y_smooth[1:N+T]))
temps   = DataFrame(series.values)

width   = 5
shifted = temps.shift(width - 1)
window  = shifted.rolling(window=width)
dataframe = concat([window.min(), window.max(), window.mean(), temps], axis=1)
#dataframe.columns = ["min", "max", "mean", "t+1"]



#plt.plot(time[1:N], np.abs(y_smooth[1:N]), '-r')
#plt.grid()
#plt.show()


input_file1 = open(r'C:\Users\Mortagetti\Desktop\Notes for School\Summer_Research\gear1_2\MPU6500 Acceleration Sensor.csv', "r")

time1 = []
x1    = []
y1    = []
z1    = []

for line in input_file1:
    if len(line) > 0:

        parts = line.split(',')

        time1.append(parts[0])
        x1.append(parts[1])
        y1.append(parts[2])
        z1.append(parts[3])

x1 = np.array(x1)
y1 = np.array(y1)
z1 = np.array(z1)

# Number of sample points
N = 1000

acceleration_data1 = np.concatenate((x1,y1,z1), axis=0)

#Normalizing the acceleration data
#acceleration_data = acceleration_data.reshape(-1,1)
#normalized_data = normalize(acceleration_data).ravel()

#fisher_score()
# FT without smoothing
ft = fft(acceleration_data1)



# FT with smoothing
rft1 = rfft(acceleration_data1)
y_smooth1 = irfft(rft1)


# Windowing through pandas found on https://machinelearningmastery.com/basic-feature-engineering-time-series-data-python/
series1  = Series(np.abs(y_smooth1[1:N+T]))
temps1   = DataFrame(series1.values)

width1   = 5
shifted1 = temps1.shift(width1 - 1)
window1  = shifted1.rolling(window=width1)
dataframe1 = concat([window1.min(), window1.max(), window1.mean(), temps1], axis=1)

######################################################

# Bi-LSTM





#####################################################

# SVM
dataframe = dataframe.drop([0,1,2,3,4,5,6,7], axis=0)
dataframe1 = dataframe1.drop([0,1,2,3,4,5,6,7], axis=0)

X_train = concat([dataframe.iloc[0:N], dataframe1.iloc[0:N]])
Y_train = [0]*len(dataframe.iloc[0:N]) + [1]*len(dataframe1.iloc[0:N])

X_test = concat([dataframe.iloc[N:N+T], dataframe1.iloc[N:N+T]])
Y_test = [0]*len(dataframe.iloc[N:N+T]) + [1]*len(dataframe1.iloc[N:N+T])
clf = svm.SVC(C=0.6, gamma='scale')
clf.fit(X_train,Y_train)

true_count = 0

for output in range(0,len(X_test)):
    if clf.predict(X_test)[output] == Y_test[output]:
        true_count += 1

print(true_count/len(X_test))

print(sklearn.metrics.mean_squared_error(Y_test, clf.predict(X_test)))
#####################################################

# DAG SVM


