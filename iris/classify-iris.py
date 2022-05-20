# Felix Zailskas (S3918270)
# Introduction to Neural Networks 2021

import sys
import numpy as np
import pandas as pd
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning

#----------------------------0 SET UP----------------------------#

# First I define the helper functions that I used in the Lab exercises

def calcAccuracy(df_test, col1, col2):
    return np.where(df_test[col1] == df_test[col2], 1, 0).sum() / len(df_test)

def mostFreqClass(df_test, col):
    return df_test[col].mode()[0]

def mostFreqPred(df_test, col):
    return np.full((len(df_test), 1), mostFreqClass(df_test, col))

# Now I deliberately suppress a warning for readability and set the seed for the numpy random
warnings.simplefilter("ignore", category=ConvergenceWarning)
np.random.seed(12)

print("\n#----------------------------1 LOADING DATA----------------------------#")
# Loading the data from the csv file
print("Loading data from the .csv file...")
df_data = pd.read_csv("iris.csv", sep=";")
print("Following data has been loaded:")
print(df_data)

# Splitting data into training and test set
print("\nI will now split the data into test and training set.\n"
      "For this I will use two thresholds to compare the results.\n"
      "One threshold will be 0.75 and the other will be 0.50, meaning\n"
      "that in the first split about 75% of the data will be used for\n"
      "training and in the second only around 50% will be used for\n"
      "training.")

thresholdLarge = 0.75
thresholdSmall = 0.5

maskLarge = np.random.random(len(df_data)) < thresholdLarge
maskSmall = np.random.random(len(df_data)) < thresholdSmall

df_train_large = df_data[maskLarge].copy()
df_test_large = df_data[~maskLarge].copy()

df_train_small = df_data[maskSmall].copy()
df_test_small = df_data[~maskSmall].copy()

# Verify the proportion of data samples in the two sets is as expected:
print("\nLet us inspect the proportion of the training and test set for both thresholds.\n"
      "Threshold 0.75:\n"
      "Training: {}\tTest: {}"
      "\nThreshold 0.50:\n"
      "Training: {}\tTest: {}".format(len(df_train_large), len(df_test_large), len(df_train_small), len(df_test_small)))

print("\n#----------------------------2 TARGET AND FEATURE SELECTION----------------------------#")
print("As we want to predict the species of the iris flowers, we will use the 'Class' attribute\n"
      "as the target. All remaining attributes of the flowers will be used as the features.")

# setting target and features
target = ["Class"]
features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

print("\nI will now slice the dataframe based on the defined target and features to make training\n"
      "and test sets.")

# large threshold
X_train_large = df_train_large[features].to_numpy()
y_train_large = df_train_large[target].to_numpy()

X_test_large = df_test_large[features].to_numpy()
y_test_large = df_test_large[target].to_numpy()

y_train_large = y_train_large.reshape((-1,))
y_test_large = y_test_large.reshape((-1,))

# small threshold
X_train_small = df_train_small[features].to_numpy()
y_train_small = df_train_small[target].to_numpy()

X_test_small = df_test_small[features].to_numpy()
y_test_small = df_test_small[target].to_numpy()

y_train_small = y_train_small.reshape((-1,))
y_test_small = y_test_small.reshape((-1,))

print("\n#----------------------------3 BUILDING THE CLASSIFIER----------------------------#")
print("To investigate which value for max-iter might be the best for the logistic regression\n"
      "model I will initialize 3 models for each threshold. These models will have values\n"
      "100, 20 and 5 as the value for max-iter. Later we will compare the results.")
my_model_large_100 = LogisticRegression(max_iter=100)
my_model_large_20 = LogisticRegression(max_iter=20)
my_model_large_5 = LogisticRegression(max_iter=5)

my_model_small_100 = LogisticRegression(max_iter=100)
my_model_small_20 = LogisticRegression(max_iter=20)
my_model_small_5 = LogisticRegression(max_iter=5)

print("\n#----------------------------4 TRAINING THE CLASSIFIER----------------------------#")
print("To train the models I will fit them to the specific training data I have created for them.")
my_model_large_100.fit(X_train_large, y_train_large)
my_model_large_20.fit(X_train_large, y_train_large)
my_model_large_5.fit(X_train_large, y_train_large)

my_model_small_100.fit(X_train_small, y_train_small)
my_model_small_20.fit(X_train_small, y_train_small)
my_model_small_5.fit(X_train_small, y_train_small)

print("\n#----------------------------5 USING THE CLASSIFIER. A.K.A PREDICTION----------------------------#")
print("We can now use the predictors to predict the species of iris flower based on the sepal\n"
      "and petal size.")
predictions_large_100 = my_model_large_100.predict(X_test_large)
predictions_large_20 = my_model_large_20.predict(X_test_large)
predictions_large_5 = my_model_large_5.predict(X_test_large)
df_test_large['LogReg_pred_100'] = predictions_large_100
df_test_large['LogReg_pred_20'] = predictions_large_20
df_test_large['LogReg_pred_5'] = predictions_large_5

predictions_small_100 = my_model_small_100.predict(X_test_small)
predictions_small_20 = my_model_small_20.predict(X_test_small)
predictions_small_5 = my_model_small_5.predict(X_test_small)
df_test_small['LogReg_pred_100'] = predictions_small_100
df_test_small['LogReg_pred_20'] = predictions_small_20
df_test_small['LogReg_pred_5'] = predictions_small_5

print("Let us now investigate the now updated data frames.\n"
      "Threshold 0.75:")
print(df_test_large)
print("Threshold 0.50:")
print(df_test_small)

print("\n#----------------------------6 EVALUATION----------------------------#")
print("To evaluate the accuracy of the predictors I will determine the ratio\n"
      "of correct predictions for both of them and compare them to a baseline."
      "I will store all these values in a vector for easier comparison.")

# adding the baseline predictions
df_test_large["mostFreqPred"] = mostFreqPred(df_test_large, "Class")
df_test_small["mostFreqPred"] = mostFreqPred(df_test_small, "Class")

# storing accuracy scores
model_scores_large = {}
model_scores_small = {}

model_scores_large['Logistic Regression Accuracy max-iter=100'] = calcAccuracy(df_test_large, "Class", "LogReg_pred_100")
model_scores_large['Logistic Regression Accuracy max-iter=20'] = calcAccuracy(df_test_large, "Class", "LogReg_pred_20")
model_scores_large['Logistic Regression Accuracy max-iter=5'] = calcAccuracy(df_test_large, "Class", "LogReg_pred_5")
model_scores_large['Most Frequent Prediction Accuracy'] = calcAccuracy(df_test_large, "Class", "mostFreqPred")

model_scores_small['Logistic Regression Accuracy max-iter=100'] = calcAccuracy(df_test_small, "Class", "LogReg_pred_100")
model_scores_small['Logistic Regression Accuracy max-iter=20'] = calcAccuracy(df_test_small, "Class", "LogReg_pred_20")
model_scores_small['Logistic Regression Accuracy max-iter=5'] = calcAccuracy(df_test_small, "Class", "LogReg_pred_5")
model_scores_small['Most Frequent Prediction Accuracy'] = calcAccuracy(df_test_large, "Class", "mostFreqPred")

print("\nNow let us investigate the accuracies of the different prediction models.")
print("Threshold 0.75:")
print('Logistic Regression Accuracy max-iter=100: {}'.format(model_scores_large.get('Logistic Regression Accuracy max-iter=100')))
print('Logistic Regression Accuracy max-iter=20: {}'.format(model_scores_large.get('Logistic Regression Accuracy max-iter=20')))
print('Logistic Regression Accuracy max-iter=5: {}'.format(model_scores_large.get('Logistic Regression Accuracy max-iter=5')))
print('Most Frequent Prediction Accuracy: {}'.format(model_scores_large.get('Most Frequent Prediction Accuracy')))
print("Threshold 0.50:")
print('Logistic Regression Accuracy max-iter=100: {}'.format(model_scores_small.get('Logistic Regression Accuracy max-iter=100')))
print('Logistic Regression Accuracy max-iter=20: {}'.format(model_scores_small.get('Logistic Regression Accuracy max-iter=20')))
print('Logistic Regression Accuracy max-iter=5: {}'.format(model_scores_small.get('Logistic Regression Accuracy max-iter=5')))
print('Most Frequent Prediction Accuracy: {}'.format(model_scores_small.get('Most Frequent Prediction Accuracy')))

print("\n#----------------------------7 CONCLUSIONS----------------------------#")
print("I would expect the following things to be observed in the data:")
print("\t1. The higher the max-iter value, the higher the model accuracy.\n"
      "\t2. The model with more training data performs better on the test data.\n"
      "\t3. The logistic regression model performs better on the test data than\n"
      "\t   the most frequent prediction model for both thresholds and all\n"
      "\t   max-iter values.")
print("1.:\n"
      "For both threshold = 0.75 and threshold = 0.50 it can be seen that\n"
      "the accuracy of the model with max-iter=20 has the highest prediction\n"
      "accuracy. Furthermore we can see that the model with max-iter=5 has\n"
      "the lowest prediction accuracy for both threshold values. Also it \n"
      "should be noted that the difference in prediction accuracy between\n"
      "the two larger values for max-iter (100 and 20) and the lowest value\n"
      "for max-iter (5) is larger than that that between the values 100 and\n"
      "20 for max-iter. The difference between max-iter=100 and max-iter=20\n"
      "for both threshold values is ~0.03, while that between max-iter=20\n"
      "and max-iter=5 is ~0.25 for both threshold values. From the observed\n"
      "results I will conclude that a max-iter value of >=20 is sufficient.\n"
      "This is largely in accordance with the prediction I made before.")
print("2.:\n"
      "We can observe that the prediction accuracies for both threshold values\n"
      "are quite similar. This is consistent with all values for max-iter. The\n"
      "largest deviation between accuracies with the same value for max-iter value is\n"
      "~0.02. In regard to my prediction it might be the case that the difference\n"
      "of the threshold values might not be large enough to detect a significant\n"
      "difference in prediction accuracy. However, the obtained results suggest\n"
      "no difference in prediction accuracy for the tested thresholds.")
print("3.:\n"
      "This prediction is supported by the observed data. For both thresholds the\n"
      "prediction accuracy for the most frequent prediction model is 0.4. This\n"
      "amounts to a difference in prediction accuracy of ~0.35 compared with the\n"
      "worst prediction accuracy of the logistic regression model for both threshold\n"
      "values. Furthermore, compared to the best prediction accuracy of the logistic\n"
      "regression model a difference in prediction accuracy of ~0.6 can be observed.")
print("\nFinally it must be noted that the data set only contained 150 entries\n"
      "and the results might significantly change when changing the used seed.\n"
      "Therefore further tests could be performed using other seeds and different\n"
      "data sets to solidify the found results.")
print("\n"
      "-Report written by Felix Zailskas (S3918270)\n")