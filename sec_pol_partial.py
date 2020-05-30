import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


# importing Dataset (known attack signatures)
def importdata():
    balance_data = pd.read_csv("/home/anand/Dropbox/projects/thesis/smart_sdn/dataset/Wednesday-workingHours.pcap_ISCX_2class.csv",sep=',', header=0)

    # removing na and normalizing with mean
    balance_data.fillna(balance_data.mean())

    #for partial signatures (limited feature set)
    feature_cols = ['Flow Bytes/s',' Flow Packets/s', ' Label']
    balance_data = balance_data[feature_cols]  # Features

    # Printing the dataswet shape
    print("Dataset Length: ", len(balance_data))
    print("Dataset Shape: ", balance_data.shape)
    print("Column Names:", balance_data.columns)

    # Printing the dataset obseravtions
    print("Dataset: ", balance_data.head())



    return balance_data


# split the dataset
def splitdataset(balance_data):
    # Separating the target variable
    X = balance_data.values[:, 0:1]
    Y = balance_data.values[:, 2]

    #normalizing nan values to max float32
    X = np.nan_to_num(X.astype(np.float32))

    # #transforming Y to categorical label (0,1)
    # le = preprocessing.LabelEncoder()
    # le.fit(["BENIGN", "Malicious"])
    # Y = le.transform(Y)

    # Splitting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=1)

    print("Splitting complete")

    if(balance_data.isna == True):
        print("error: nan infinity in the dataset")

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
    #print(y_pred)
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

    # Prediction
    y_pred = prediction(X_test, dec_tree)
    cal_accuracy(y_test, y_pred)

# Calling main function
if __name__ == "__main__":
    main()
