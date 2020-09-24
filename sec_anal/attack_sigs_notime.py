import numpy as np
import pandas as pd
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn import preprocessing
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding
import random
from tensorflow import keras
import matplotlib.pyplot as plt


# importing Dataset (from pcap parser: known attacks)
def importdata():
    # balance_data = pd.read_csv("/home/anand/Dropbox/projects/thesis/smart_sdn/sec_anal/state_based/test2_new_train.csv",sep=',', header=0)
    #
    test_data = pd.read_csv("/home/anand/Dropbox/projects/thesis/smart_sdn/sec_anal/state_based/test2_new_test.csv",sep=',', header=0)

    balance_data = pd.read_csv("/home/anand/PycharmProjects/mininet_backend/pcaps/all_attacks.csv",sep=',', header=0)

    # # removing na and normalizing with mean
    # balance_data.fillna(balance_data.mean())
    #
    #for partial signatures (limited feature set)
    feature_cols = ['pckts_forward','bytes_forward', 'pckts_back', 'bytes_back', 'label']
    balance_data = balance_data[feature_cols]  # Features
    test_data = test_data[feature_cols]

    # Printing the dataswet shape
    print("Dataset Length: ", len(balance_data))
    print("Dataset Shape: ", balance_data.shape)
    print("Column Names:", balance_data.columns)

    # Printing the dataset obseravtions
    print("Train Dataset: ", balance_data.head())
    #print("Test Dataset: ", test_data.head())

    return balance_data, test_data

# split the dataset
def splitdataset(balance_data):
    # Separating the target variable
    X = balance_data.values[:, 0:4]
    Y = balance_data.values[:, 4]

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


def train_rl(X_train,y_train):
    reconstructed_model = keras.models.load_model("../rl_models/rl_model_v14")
    return reconstructed_model

def train_forest(X_train, y_train):
    # Creating the RF classifier object
    dec_tree = RandomForestClassifier()

    # Performing training
    dec_tree.fit(X_train, y_train)
    print("Training complete")

    #save dec tree model
    filename = 'rforest.sav'
    pickle.dump(dec_tree, open(filename, 'wb'))
    return dec_tree

def train_dtree(X_train, y_train):

    # Creating the Decision Tree classifier object
    dec_tree = DecisionTreeClassifier()

    # Performing training
    dec_tree.fit(X_train, y_train)
    print("Training complete")

    #save dec tree model
    filename = 'partial_sig.sav'
    pickle.dump(dec_tree, open(filename, 'wb'))
    return dec_tree

def train_dnn(X_train,y_train):

    #DNN
    model = Sequential()
    model.add(Dense(128, input_dim=4, activation='relu'))
    model.add(Dense(128, activation='relu'))
    #model.add(Dense(1024, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam(lr=0.001,decay=0.001))

    #minibatch = random.sample((X_train,y_train),50)

    model.fit(X_train,y_train,epochs=50)
    return model

# Function to make predictions
def prediction_forest(X_test, clf_object):
    # Predicton on test with giniIndex
    y_pred = clf_object.predict(X_test)
    print("Predicted values:")
    #print(y_pred)
    return y_pred

def prediction_dnn(X_test, clf_object,y_orig):
    # Predicton on test with giniIndex
    y_pred = clf_object.predict(X_test)
    print(X_test)
    print("Predicted values:")
    #print(y_pred, y_orig)
    #print(np.argmax(y_pred,axis=1))
    #print(np.argmax(y_pred[0]))
    # y_pred2 = clf_object.predict(np.array([[399, 45899, 2, 112], [9, 2, 5, 1],[13, 2155, 118 ,126781], [3, 156, 3, 156]]))
    # print("prediction is", y_pred2)

    #return y_pred
    return np.argmax(y_pred,axis=1)

def prediction_rl(X_test,clf_object):
    # Predicton on test with giniIndex
    y_pred = clf_object.predict(X_test)
    print("Predicted values:")
    print(y_pred)
    print(np.argmax(y_pred))
    print(np.argmax(y_pred, axis=1))

    y_pred2 = clf_object.predict((0,0,0,0))
    print("prediction is",y_pred2)

    # return y_pred
    return np.argmax(y_pred, axis=1)

# Function to calculate accuracy
def cal_accuracy(y_test, y_pred):
    print("Confusion Matrix: ",
          confusion_matrix(y_test, y_pred))

    print("Accuracy : ",
          accuracy_score(y_test, y_pred) * 100)

    print("Report : ",
          classification_report(y_test, y_pred))

def cal_timing(y_test,y_pred, yq_values):

    N = len(y_test)
    counter = 0
    o_test = []
    o_pred = []
    output = 0
    output_pred = 0
    output_count = 0

    while(counter < N):
        if (counter % 60) == 0:
            # print("session no", (counter // 10),":",output_pred,output)
            o_test.append(output)
            o_pred.append(output_pred)
            output = 0
            output_pred = 0
            output_count = 0

        if(y_test[counter] == 1):
            output_count += 1
            if(output_count >= 30):
                output = 1

        if( y_pred[counter] == 1 ):#and yq_values[counter][0] < 0 #and yq_values[counter][0] < 500 and yq_values[counter][1] > 600 #and yq_values[counter][0] < 3000 and yq_values[counter][1] > 4400
            output_pred = 1
            #print(yq_values[counter])

        counter +=1

    cal_accuracy(o_test, o_pred)


def dnn_scores(X_train,y_train,X_test,y_test,dec_tree):
    pred_train = dec_tree.predict(X_train)
    scores = dec_tree.evaluate(X_train, y_train, verbose=0)
    print('Accuracy on training data: {}% \n Error on training data: {}'.format(scores[1], 1 - scores[1]))

    pred_test = dec_tree.predict(X_test)
    scores2 = dec_tree.evaluate(X_test, y_test, verbose=0)
    print('Accuracy on test data: {}% \n Error on test data: {}'.format(scores2[1], 1 - scores2[1]))

def analyse_rl(y_test,y_pred, yq_values):

    N = len(y_test)
    counter = 0


    x = range(N)
    # avg_benign = sum(yq_values[:,0])/N
    # avg_mal = sum(yq_values[:, 1]) / N

    plt.plot(x,yq_values[:,0], label = "benign")
    plt.plot(x,yq_values[:,1], label = "mal")
    plt.plot(x,y_test[:]*100)

    plt.show()

    # while(counter < N):
    #
    #     if(y_test[counter] == 1): #malicious
    #         print(yq_values[counter])
    #     counter +=1


# Driver code
def main():
    # Building Phase
    data, test_data = importdata()
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data)

    # #IF testing
    # X_test = test_data.values[:, 0:4]
    # y_test = test_data.values[:, 4]

    le = LabelEncoder()
    le.fit(y_train)
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)

    y_orig = y_test
    #changing dims for categorical features
    y_test = to_categorical(y_test)
    y_train = to_categorical(y_train)

    print(y_train)
    print(y_orig)

    dec_tree = train_forest(X_train,y_train)
    #dec_tree = train_dnn(X_train,y_train)

    # Operational Phase
    print("Results:")

    # # #adding noise for robustness testing
    noise = np.random.normal(0, 10, X_test.shape)
    X_test += noise.round()

    #for dnn only
    #dnn_scores(X_train,y_train,X_test,y_test,dec_tree)

    yq_values = dec_tree.predict(X_test)

    #print(X_test)
    # Prediction
    y_pred = prediction_dnn(X_test, dec_tree, y_orig)
    #y_pred = prediction_forest(X_test, dec_tree)
    cal_accuracy(y_orig, y_pred)
    cal_timing(y_orig,y_pred, yq_values)
    analyse_rl(y_orig,y_pred, yq_values)

# Calling main function
if __name__ == "__main__":
    main()
