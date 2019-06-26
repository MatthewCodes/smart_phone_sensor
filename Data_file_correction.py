#from skfeature.function.similarity_based.fisher_score import fisher_score
from scipy.fftpack import fft, rfft, irfft
import numpy as np
import keras
from sklearn import svm
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier
import sklearn
from pandas import Series
from pandas import DataFrame
from pandas import concat
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import cross_val_score
# Fourier implementation from https://docs.scipy.org/doc/scipy/reference/tutorial/fftpack.html

#input_file = open('/mnt/c/Users/Mortagetti/Desktop/Sample_gyroscope_t1.txt')

def get_data(folder1, folder2, csv_file):

    incomplete_path = r'C:\Users\Mortagetti\Desktop\Notes for School\Summer_Research'
    complete_path = incomplete_path + "\\" + folder1 + "\\" + csv_file
    input_file = open(complete_path, "r")

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

    ## Feature engineering section
    # Windowing through pandas found on https://machinelearningmastery.com/basic-feature-engineering-time-series-data-python/
    series  = Series(np.abs(y_smooth[1:N+T]))
    temps   = DataFrame(series.values)

    width   = 5
    shifted = temps.shift(width - 1)
    window  = shifted.rolling(window=width)
    dataframe = concat([window.min(), window.max(), window.mean(), temps], axis=1)
    #dataframe.columns = ["min", "max", "mean", "t+1"]

    # plt.plot(time[1:N], np.abs(y_smooth[1:N]), '-r')
    # plt.grid()
    # plt.show()

    incomplete_path = r'C:\Users\Mortagetti\Desktop\Notes for School\Summer_Research'
    complete_path = incomplete_path + "\\" + folder2 + "\\" + csv_file
    input_file1 = open(complete_path, "r")
    #input_file1 = open(r'C:\Users\Mortagetti\Desktop\Notes for School\Summer_Research\gear1\MPU6500 Acceleration Sensor.csv', "r")

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

    # plt.plot(time[1:N], np.abs(y_smooth1[1:N]), '-r')
    # plt.grid()
    # plt.show()

    return [dataframe, dataframe1]


#####################################################
# Training and Testing data generation
print('Everyting loaded correctly')
acceleration_data = get_data('gear7_2', 'gear1_2', csv_file='MPU6500 Acceleration Sensor.csv')
gyroscope_data    = get_data('gear7_2', 'gear1_2', csv_file='MPU6500 Gyroscope Sensor.csv')
magnetometer_data = get_data('gear7_2', 'gear1_2', csv_file='YAS537 Magnetic Sensor.csv')

# These numbers were selected to keep the data even for batches
N = 1000
T = 240


def get_frames(acceleration, gyroscope, magnetometer):
    acceleration_dataframe  = acceleration[0]
    acceleration_dataframe1 = acceleration[1]
    acceleration_dataframe  = acceleration_dataframe.drop([0,1,2,3,4,5,6,7], axis=0)
    acceleration_dataframe1 = acceleration_dataframe1.drop([0,1,2,3,4,5,6,7], axis=0)

    gyroscope_dataframe  = gyroscope[0]
    gyroscope_dataframe1 = gyroscope[1]
    gyroscope_dataframe  = gyroscope_dataframe.drop([0,1,2,3,4,5,6,7], axis=0)
    gyroscope_dataframe1 = gyroscope_dataframe1.drop([0,1,2,3,4,5,6,7], axis=0)

    magnetometer_dataframe  = magnetometer[0]
    magnetometer_dataframe1 = magnetometer[1]
    magnetometer_dataframe  = magnetometer_dataframe.drop([0,1,2,3,4,5,6,7], axis=0)
    magnetometer_dataframe1 = magnetometer_dataframe1.drop([0,1,2,3,4,5,6,7], axis=0)

    return [acceleration_dataframe, acceleration_dataframe1, gyroscope_dataframe, gyroscope_dataframe1,
            magnetometer_dataframe, magnetometer_dataframe1]


[acceleration_dataframe, acceleration_dataframe1, gyroscope_dataframe, gyroscope_dataframe1, magnetometer_dataframe,
magnetometer_dataframe1] = get_frames(acceleration_data,gyroscope_data,magnetometer_data)


# This approach is splitting the data then separately choosing the best features in feature selection
# another approach that might increase accuracy is by selecting the features first, then splitting the data

X_train0 = concat([acceleration_dataframe.iloc[0:N], gyroscope_dataframe.iloc[0:N], magnetometer_dataframe.iloc[0:N]],
                 axis=1)

X_train1 = concat([acceleration_dataframe1.iloc[0:N], gyroscope_dataframe1.iloc[0:N], magnetometer_dataframe1.iloc[0:N]],
                   axis=1)

X_train = concat([X_train0, X_train1])
Y_train = np.array([0]*len(acceleration_dataframe.iloc[0:N]) + [1]*len(acceleration_dataframe1.iloc[0:N]))

X_test0 = concat([acceleration_dataframe.iloc[N:N+T], gyroscope_dataframe.iloc[N:N+T], magnetometer_dataframe.iloc[N:N+T]],
                 axis=1)

X_test1 = concat([acceleration_dataframe1.iloc[N:N+T], gyroscope_dataframe1.iloc[N:N+T], magnetometer_dataframe1.iloc[N:N+T]],
                   axis=1)

X_test = concat([X_test0, X_test1])
Y_test = np.array([0]*len(acceleration_dataframe.iloc[N:N+T]) + [1]*len(acceleration_dataframe.iloc[N:N+T]))


# Univariate Selection feature selection found on https://machinelearningmastery.com/feature-selection-machine-learning-python/

# feature extraction for Training
test = SelectKBest(score_func=chi2, k=8)
fit = test.fit(X_train, Y_train)
# summarize scores
np.set_printoptions(precision=3)
X_train = fit.transform(X_train)

# feature extraction for Testing
test = SelectKBest(score_func=chi2, k=8)
fit = test.fit(X_test, Y_test)
# summarize scores
np.set_printoptions(precision=3)
X_test = fit.transform(X_test)

data1 = np.column_stack((X_train, Y_train))
data1 = DataFrame(data1)
data1.to_csv('labeled_training.csv')


data2 = np.column_stack((X_test, Y_test))
data2 = DataFrame(data2)
data2.to_csv('labeled_testing.csv')
print('labeled')


#####################################################

# Bi-LSTM
def generate_batches(training_size, batch_size):
    batches = []
    for i in range(0,training_size,batch_size):
        batch = [i,i+batch_size]
        batches.append(batch)
    return batches


n_timesteps = 10

model = Sequential()

LSTM_model = keras.layers.LSTM(32, input_shape=(9,8), activation='sigmoid', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=True, return_state=False, go_backwards=False, stateful=False, unroll=False)

model.add(keras.layers.Bidirectional(LSTM_model))
model.add(keras.layers.TimeDistributed(Dense(8,activation='tanh'))) # Look into what this is, might need to be set to 9
# model.add(keras.layers.Flatten())
# model.add(Dense(5, activation='softmax'))

model.compile(loss='mean_squared_error', optimizer='adam')
for epoch in range(50):
    batches = generate_batches(len(X_train), 20)

    for batch in batches:
        X = X_train[batch[0]:batch[1]]
        Y = Y_train[batch[0]:batch[1]]

        data = np.column_stack((X,Y))

        X = data[:,:-1]
        Y = data[:,1:]

        X = X.reshape((20, 1, 8))
        Y = Y.reshape((20, 1, 8))

        model.train_on_batch(X, Y)
    print(epoch)

batches = generate_batches(len(X_test), 20)
batch = batches[0]

X = X_test[batch[0]:batch[1]]
Y = Y_test[batch[0]:batch[1]]

data = np.column_stack((X,Y))
X = data[:,:-1]
Y = data[:,1:]

X = X.reshape((20, 1, 8))
Y = Y.reshape((20, 1, 8))
loss_metrics = model.evaluate(X, Y)

print(loss_metrics)
######################################################
# Best so far is 75% with C = 1.9 gamma = scale and a poly kernel function
# SVM
clf = svm.SVC(C=1.9, gamma='scale', kernel='poly')
clf.fit(X_train,Y_train)


print(clf.score(X_test, Y_test))
print(cross_val_score(clf, X_test, Y_test, cv=3))

#####################################################

# DAG SVM See matlab file

#####################################################

# Random Forest

clf = RandomForestClassifier(n_estimators=15, criterion='entropy', max_depth=3, min_samples_split=20, bootstrap=True, random_state=0)

clf.fit(X_train, Y_train)
print(clf.feature_importances_)

print(clf.score(X_test, Y_test))
print(cross_val_score(clf, X_test, Y_test, cv=3))
#####################################################
