#-------------------------------------------------------------------------
# AUTHOR: Arze Lu
# FILENAME: knn.py
# SPECIFICATION: supervised weather training
# FOR: CS 5990 - Assignment #4
# TIME SPENT: 5 hours
#-------------------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# defining the hyperparameter values of KNN
k_values = [i for i in range(1, 20)]
p_values = [1, 2] 
w_values = ['uniform', 'distance']

# reading the training data
train_data = pd.read_csv("weather_training.csv")

# reading the test data
test_data = pd.read_csv("weather_test.csv")

#hint: to convert values to float while reading them -> np.array(df.values)[:,-1].astype('f')
X_train = np.array(train_data.iloc[:, 1:-1]).astype('float')
y_train = np.array(train_data.iloc[:, -1]).astype('float')
X_test = np.array(test_data.iloc[:, 1:-1]).astype('float')
y_test = np.array(test_data.iloc[:, -1]).astype('float')

# 11-class discretization setup
classes = [i for i in range(-22, 40, 6)]

y_test = np.digitize(y_test, classes)
y_train = np.digitize(y_train, classes)

bin_midpoints = [(classes[i] + classes[i+1])/2 for i in range(len(classes)-1)]

#loop over the hyperparameter values (k, p, and w) ok KNN
#--> add your Python code here
best_accuracy = 0

for k in k_values:
    for p in p_values:
        for w in w_values:
            #fitting the knn to the data
            #--> add your Python code here
            clf = KNeighborsClassifier(n_neighbors=k, p=p, weights=w)
            clf.fit(X_train, y_train)

            #make the KNN prediction for each test sample and start computing its accuracy
            #hint: to iterate over two collections simultaneously, use zip()
            #Example. for (x_testSample, y_testSample) in zip(X_test, y_test):
            #to make a prediction do: clf.predict([x_testSample])
            #the prediction should be considered correct if the output value is [-15%,+15%] distant from the real output values.
            #to calculate the % difference between the prediction and the real output values use: 100*(|predicted_value - real_value|)/real_value))
            #--> add your Python code here

            #check if the calculated accuracy is higher than the previously one calculated. If so, update the highest accuracy and print it together
            #with the KNN hyperparameters. Example: "Highest KNN accuracy so far: 0.92, Parameters: k=1, p=2, w= 'uniform'"
            #--> add your Python code here
            correct = 0
            total = len(X_test)

            for x_test_sample, y_test_sample in zip(X_test, y_test):
                y_pred = clf.predict([x_test_sample])[0]

                # Convert prediction bin index back to original value
                y_pred = bin_midpoints[y_pred - 1]
                real_value = bin_midpoints[y_test_sample - 1]

                # Check if prediction is within ±15% of the true value
                percent_diff = 100 * abs(y_pred - real_value) / abs(real_value)
                if percent_diff <= 15:
                    correct += 1

            accuracy = correct / total

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                print(f"Highest KNN accuracy so far: {accuracy:.5f}")
                print(f"Parameters: k = {k}, p = {p}, weight = {w}\n")