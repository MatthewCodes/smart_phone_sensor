# must do binning, data aug, and try different fft params

#from skfeature.function.similarity_based.fisher_score import fisher_score
from scipy.fftpack import fft, rfft, irfft

from scipy import fftpack
import numpy as np
import matplotlib.pyplot as plt
import keras
from sklearn import svm
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier
from pandas import Series
from pandas import DataFrame
from pandas import concat
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile, chi2
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import cross_val_score


def get_data(folder1, csv_file):

    incomplete_path = r'..'
    complete_path = incomplete_path + "\\" + folder1 + "\\" + csv_file
    print(complete_path)
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
    data = np.concatenate((x, y, z), axis=0)

    f = 10

    # Number of sample points for training
    N = int(len(data) * .75)

    # The sample points for testing
    T = int(len(data) * .25)

    #Normalizing the acceleration data
    data = data.reshape(-1,1)
    data = normalize(data, axis=0).ravel()


    # plot the raw timeseries data
    """
    N1 = len(acceleration_data)
    T1 = 1.0 /2000
    x = np.linspace(0.0, N1*T1, N1)
    yf = rfft(acceleration_data)
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
    plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
    plt.show()
    """

    # Fourier implementation from https://docs.scipy.org/doc/scipy/reference/tutorial/fftpack.html
    rft = rfft(data)
    y_smooth = irfft(rft)

    ## Feature engineering section
    # Windowing through pandas found on https://machinelearningmastery.com/basic-feature-engineering-time-series-data-python/
    series  = Series(np.abs(y_smooth[1:N+T]))
    temps   = DataFrame(series.values)
    width   = 5
    shifted = temps.shift(width - 1)
    window  = shifted.rolling(window=width)
    dataframe = concat([window.min(), window.max(), window.mean(), temps], axis=1)

    # Plot the bins for the timeseries data
    """
    fig, ax = plt.subplots()
    f_s = 100
    freqs = fftpack.fftfreq(len(acceleration_data)) * f_s
    ax.stem(freqs, np.abs(y_smooth))
    ax.set_xlim(-f_s / 2, f_s / 2)
    ax.set_ylim(-5, 3000)
    series.plot.hist(grid=True, bins=20)
    
    plt.ylim(top=16000)
    plt.plot(freqs, np.abs(y_smooth), '-r')
    plt.grid()
    plt.show()
    """
    # incomplete_path = r'C:\Users\Mortagetti\Desktop\Notes for School\Summer_Research'
    # complete_path = incomplete_path + "\\" + folder2 + "\\" + csv_file
    # input_file1 = open(complete_path, "r")
    # #input_file1 = open(r'C:\Users\Mortagetti\Desktop\Notes for School\Summer_Research\gear1\MPU6500 Acceleration Sensor.csv', "r")
    #
    # time1 = []
    # x1    = []
    # y1    = []
    # z1    = []
    #
    # for line in input_file1:
    #     if len(line) > 0:
    #
    #         parts = line.split(',')
    #
    #         time1.append(parts[0])
    #         x1.append(parts[1])
    #         y1.append(parts[2])
    #         z1.append(parts[3])
    #
    # x1 = np.array(x1)
    # y1 = np.array(y1)
    # z1 = np.array(z1)
    #
    # # Number of sample points
    #
    #
    # acceleration_data1 = np.concatenate((x1,y1,z1), axis=0)
    # # Number of sample points for training
    # N = int(len(acceleration_data1) * .75)
    #
    # # The sample points for testing
    # T = int(len(acceleration_data1) * .25)
    #
    #
    # N1 = len(acceleration_data1)
    # T1 = 1.0 /2000
    # x = np.linspace(0.0, N1*T1, N1)
    # yf = rfft(acceleration_data1)
    # xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
    # plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
    # plt.show()
    #
    #
    #
    # #Normalizing the acceleration data
    # #acceleration_data = acceleration_data.reshape(-1,1)
    # #normalized_data = normalize(acceleration_data).ravel()
    #
    # #fisher_score()
    # # FT without smoothing
    # ft = fft(acceleration_data1)
    #
    #
    #
    # # FT with smoothing
    # rft1 = rfft(acceleration_data1)
    # freqs1 = fftpack.fftfreq(len(acceleration_data1)) * f_s
    # y_smooth1 = irfft(rft1)
    #
    #
    # # Windowing through pandas found on https://machinelearningmastery.com/basic-feature-engineering-time-series-data-python/
    # series1  = Series(np.abs(y_smooth1[1:N+T]))
    # temps1   = DataFrame(series1.values)
    #
    # width1   = 5
    # shifted1 = temps1.shift(width1 - 1)
    # window1  = shifted1.rolling(window=width1)
    # dataframe1 = concat([window1.min(), window1.max(), window1.mean(), temps1], axis=1)
    #
    # series1.plot.hist(grid=True, bins=20)
    # #plt.plot(freqs1, np.abs(y_smooth1), '-r')
    # plt.ylim(top=16000)
    # #plt.grid()
    # plt.show()

    return [dataframe]


def generate_batches(training_size, batch_size):
    batches = []
    for i in range(0,training_size,batch_size):
        batch = [i,i+batch_size]
        if batch[1] - batch[0] == batch_size and batch[1] < training_size:
            batches.append(batch)
    return batches


def data_size_reduction(data, df):
    if df:
        min_len = np.inf
        for column in data:
            if len(data[column]) < min_len:
                min_len = len(data[column])
        return min_len

    min_len = np.inf
    for column in data:
        if len(column) < min_len:
            min_len = len(column)
    return min_len


def get_frames(acceleration, gyroscope, magnetometer):
    acceleration_dataframe  = acceleration[0]
    acceleration_dataframe  = acceleration_dataframe.drop([0,1,2,3,4,5,6,7], axis=0)
    gyroscope_dataframe  = gyroscope[0]
    gyroscope_dataframe  = gyroscope_dataframe.drop([0,1,2,3,4,5,6,7], axis=0)
    magnetometer_dataframe  = magnetometer[0]
    magnetometer_dataframe  = magnetometer_dataframe.drop([0,1,2,3,4,5,6,7], axis=0)

    return [acceleration_dataframe, gyroscope_dataframe,
            magnetometer_dataframe]

#####################################################
# Training and Testing data generation
print('Everyting loaded correctly')

#First data chunk to load
acceleration_data = get_data('gear7_2', csv_file='MPU6500 Acceleration Sensor.csv')
gyroscope_data = get_data('gear7_2', csv_file='MPU6500 Gyroscope Sensor.csv')
magnetometer_data = get_data('gear7_2', csv_file='YAS537 Magnetic Sensor.csv')


[acceleration_dataframe, gyroscope_dataframe, magnetometer_dataframe] = get_frames(acceleration_data,gyroscope_data,magnetometer_data)


# This approach is splitting the data then separately choosing the best features in feature selection
# another approach that might increase accuracy is by selecting the features first, then splitting the data

X_train = concat([acceleration_dataframe.iloc[0:int((len(acceleration_dataframe) * 0.75)/2)],
             gyroscope_dataframe.iloc[0:int((len(gyroscope_dataframe) * 0.75)/2)],
                  magnetometer_dataframe.iloc[0:int((len(magnetometer_dataframe) * 0.75)/2)]],
                 axis=1)

Y_train = [22]*data_size_reduction(X_train, True)

X_test = concat([acceleration_dataframe.iloc[int((len(acceleration_dataframe) * 0.75)/2):int((len(acceleration_dataframe) * 0.75)/2) + int((len(acceleration_dataframe) * 0.25)/2)],
                gyroscope_dataframe.iloc[int((len(gyroscope_dataframe) * 0.75)/2):int((len(gyroscope_dataframe) * 0.75)/2) + int((len(gyroscope_dataframe) * 0.25)/2)],
                magnetometer_dataframe.iloc[int((len(magnetometer_dataframe) * 0.75)/2):int((len(magnetometer_dataframe) * 0.75)/2) + int((len(magnetometer_dataframe) * 0.25)/2)]],
                axis=1)

Y_test = [22]*data_size_reduction(X_test, True)

# Naming convention used here g1_d1_1 means gear 1 with derailuer 1 (smallest) and trail 1
gears = ['1', '4', '6']
derail = ['1', '2']
folders = []

for g in gears:
    for d in derail:
        folders.append('g' + g + '_d' + d)
possible_folders = []
for f in folders:
    possible_folders.append(f + '_0')
    possible_folders.append(f + '_1')
    possible_folders.append(f + '_2')
    possible_folders.append(f + '_3')
    possible_folders.append(f + '_4')
    possible_folders.append(f + '_5')
    possible_folders.append(f + '_6')


# Used if you want to have each gear and derailleur combination be a class
all_targets = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

for folder in possible_folders:
    try:
        target = 0
        if folder[1] == '1':
            target = 0
        elif folder[1] == '4':
            target = 1
        else:
            target = 2

        acceleration_data = get_data(folder, csv_file='MPU6500 Acceleration Sensor.csv')
        gyroscope_data = get_data(folder, csv_file='MPU6500 Gyroscope Sensor.csv')
        magnetometer_data = get_data(folder, csv_file='YAS537 Magnetic Sensor.csv')

        [acceleration_dataframe, gyroscope_dataframe, magnetometer_dataframe] = get_frames(acceleration_data,
                                                                                           gyroscope_data,
                                                                                           magnetometer_data)
        X_train1 = concat(
            [acceleration_dataframe.iloc[0:int((len(acceleration_dataframe) * 0.75)/2)],
             gyroscope_dataframe.iloc[0:int((len(gyroscope_dataframe) * 0.75)/2)],
             magnetometer_dataframe.iloc[0:int((len(magnetometer_dataframe) * 0.75)/2)]],
            axis=1)

        Y_train1 = [all_targets[target]] * data_size_reduction(X_train1, True)

        X_test1 = concat([acceleration_dataframe.iloc[int((len(acceleration_dataframe) * 0.75)/2):int((len(acceleration_dataframe) * 0.75)/2) + int((len(acceleration_dataframe) * 0.25)/2)],
                          gyroscope_dataframe.iloc[int((len(gyroscope_dataframe) * 0.75)/2):int((len(gyroscope_dataframe) * 0.75)/2) + int((len(gyroscope_dataframe) * 0.25)/2)],
                         magnetometer_dataframe.iloc[int((len(magnetometer_dataframe) * 0.75)/2):int((len(magnetometer_dataframe) * 0.75)/2) + int((len(magnetometer_dataframe) * 0.25)/2)]],
                        axis=1)

        min_length = data_size_reduction(X_test1, True)
        Y_test1 = [all_targets[target]] * min_length


        # Used if you want to have each gear and derailleur combination be a class
        all_targets.pop(0)

        # Adds the new data to the existing matrix of data
        X_train = concat([X_train, X_train1], axis=0)
        Y_train = Y_train + Y_train1
        X_test = concat([X_test, X_test1], axis=0)
        Y_test = Y_test + Y_test1


    except:
        pass

Y_train = np.array(Y_train)
Y_test  = np.array(Y_test)

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)


# data augmentation
noisy_data = []
print("starting data augmentation")
X_test.columns = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"]
X_train.columns = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"]
for col in X_test:
    feature = np.array(X_test[col])
    # Used to replace nan with 0
    feature = np.nan_to_num(feature) #feature = feature[~np.isnan(feature)] use this to get rid of Nan
    # print(np.isnan(feature))
    yy = np.abs(np.random.normal(np.nanmean(feature), np.nanstd(feature), size=len(feature)))
    noisy_data.append(yy)

max = data_size_reduction(noisy_data, False)
X_test = noisy_data[0][0:max]
for i in range(1,12):
    X_test = np.column_stack((X_test, noisy_data[i][0:max]))


# Removing Nan from data
def data_clean(data):

    new_data = []
    for col in data:
        feature = np.array(data[col])
        feature = np.nan_to_num(feature)
        new_data.append(feature)
    min = data_size_reduction(new_data, False)
    data = new_data[0][0:min]
    for i in range(1, 12):
        data = np.column_stack((data, new_data[i][0:min]))
    return data


X_train = data_clean(X_train)


print("After Cleaning")
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)


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

data2 = np.column_stack((X_test, Y_test))
data2 = DataFrame(data2)


#####################################################
print("generating batches")
# Bi-LSTM

n_timesteps = 10

model = Sequential()

LSTM_model = keras.layers.LSTM(32, input_shape=(8,8), activation='sigmoid', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=True, return_state=False, go_backwards=False, stateful=False, unroll=False)

model.add(keras.layers.Bidirectional(LSTM_model))
model.add(keras.layers.TimeDistributed(Dense(8,activation='softmax'))) # Look into what this is, might need to be set to 9

model.compile(loss='mean_squared_error', optimizer='adam')
print("training Bi-LSTM")
batches = generate_batches(len(X_train), 64)
print("batches generated")
"""
for epoch in range(10):

    for batch in batches:
        X = X_train[batch[0]:batch[1]]
        Y = Y_train[batch[0]:batch[1]]

        data = np.column_stack((X,Y))

        X = data[:,:-1]
        Y = data[:,1:]

        X = X.reshape((2, 32, 8))
        Y = Y.reshape((2, 32, 8))

        model.train_on_batch(X, Y)
    print("Epoch: " + str(epoch))

batches = generate_batches(len(X_test), 64)
loss_metrics_min = []

for batch in batches:

    X = X_test[batch[0]:batch[1]]
    Y = Y_test[batch[0]:batch[1]]

    data = np.column_stack((X,Y))
    X = data[:,:-1]
    Y = data[:,1:]

    # Formatted samples, timesteps, features
    X = X.reshape((2, 32, 8))
    Y = Y.reshape((2, 32, 8))
    loss_metrics = model.evaluate(X, Y)
    loss_metrics_min.append(loss_metrics)

print(min(loss_metrics_min))
"""
######################################################
# Best so far is 23% with C = 1.9 gamma = scale and a rbf kernel function
# SVM
clf = svm.SVC(C=1.6, kernel='rbf', degree=8) #gamma'scaled'
print("training SVM")
clf.fit(X_train,Y_train)

print('Score: '+ str(clf.score(X_test, Y_test)))
print('3 Fold CV Score:')
print(cross_val_score(clf, X_test, Y_test, cv=3))

#####################################################

# DAG SVM See matlab file

#####################################################

# Random Forest

clf = RandomForestClassifier(n_estimators=15, criterion='entropy', max_depth=3, min_samples_split=20, bootstrap=True, random_state=0)
print("training Random Forest")
clf.fit(X_train, Y_train)


print('Score: '+ str(clf.score(X_test, Y_test)))
print('3 Fold CV Score:')
print(cross_val_score(clf, X_test, Y_test, cv=3))
#####################################################
