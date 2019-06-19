tr_data = readtable('../smart_phone_sensor/labeled_training.csv')

te_data = readtable('../smart_phone_sensor/labeled_testing.csv')

X_train = table2array(tr_data(:,2:9))
Y_train = table2array(tr_data(:,end))
X_test  = table2array(te_data(:,2:9))
Y_test  = table2array(te_data(:,end))
result = DAGsvm(X_train, Y_train, X_test, Y_test, 2.5, 0.5) 

