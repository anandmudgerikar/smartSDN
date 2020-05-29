import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


# importing Dataset (known attack signatures)
def importdata():
    balance_data = pd.read_csv("/home/anand/Dropbox/projects/thesis/smart_sdn/dataset/Wednesday-workingHours.pcap_ISCX_2class.csv",sep=',', header=None)

    # Printing the dataswet shape
    print("Dataset Length: ", len(balance_data))
    print("Dataset Shape: ", balance_data.shape)

    # removing na and normalizing with mean
    balance_data.fillna(balance_data.mean())

    # Printing the dataset obseravtions
    print("Dataset: ", balance_data.head())
    return balance_data


# split the dataset
def splitdataset(balance_data):
    # Separating the target variable
    X = balance_data.values[:, 0:77]
    Y = balance_data.values[:, 78]

    # Splitting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=100)

    print("Splitting complete")

    return X, Y, X_train, X_test, y_train, y_test


def train(X_train, y_train):
    # Creating the Decision Tree classifier object
    dec_tree = DecisionTreeClassifier()

    # Performing training
    dec_tree.fit(X_train, y_train)
    print("Training complete")
    return dec_tree

# Function to make predictions
def prediction(X_test, clf_object):
    # Predicton on test with giniIndex
    y_pred = clf_object.predict(X_test)
    print("Predicted values:")
    print(y_pred)
    return y_pred


# Function to calculate accuracy
def cal_accuracy(y_test, y_pred):
    print("Confusion Matrix: ",
          confusion_matrix(y_test, y_pred))

    print("Accuracy : ",
          accuracy_score(y_test, y_pred) * 100)

    print("Report : ",
          classification_report(y_test, y_pred))


# Driver code
def main():
    # Building Phase
    data = importdata()
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data)
    dec_tree = train(X_train,y_train)

    # Operational Phase
    print("Results:")

    # Prediction using gini
    y_pred = prediction(X_test, dec_tree)
    cal_accuracy(y_test, y_pred)

# Calling main function
if __name__ == "__main__":
    main()
