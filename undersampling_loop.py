import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings('ignore',category=DeprecationWarning)

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.metrics import Recall

### Functions

def gridsearch_score(x_train_data,y_train_data):
    # Find the best hyper-parameter optimizing for recall
    c_param_range = [0.01,0.1,1,10,100]
    clf = GridSearchCV(LogisticRegression(), {"C": c_param_range}, cv=5, scoring='recall')
    clf.fit(x_train_data,y_train_data)        
    return clf.best_params_["C"]

def network_builder(hidden_dimensions, input_dim):
    # Neural Network (NN) with multiple hidden layers
    # create model
    model = Sequential()
    model.add(Dense(hidden_dimensions[0], input_dim=input_dim, kernel_initializer='normal', activation='relu'))
    
    # add multiple hidden layers
    for dimension in hidden_dimensions[1:]:
        model.add(Dense(dimension, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    
    # Compile model. Use the the logarithmic loss function, and the Adam gradient optimizer.
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[ Recall() ])
    
    return model

def gridsearch_score_deep_learning(x_train_data,y_train_data, scoring="recall"):
    # Find the best hyper-parameter (hidden layer dimensions) optimizing for recall
    parameters = {
        "hidden_dimensions": [
            [5],
            [5, 5],
            [10],
            [10, 10, 10],
            [100, 10],
            [100, 10, 10],
            [100, 100, 10]
        ]
    }

    clf = GridSearchCV(KerasClassifier(build_fn=network_builder, epochs=100, batch_size=128, 
        verbose=0, input_dim=x_train_data.shape[1]), parameters, cv=5, scoring=scoring)
    clf.fit(x_train_data,y_train_data)
    return clf.best_params_["hidden_dimensions"]



### load data
data = pd.read_csv("ai4i2020.csv")

# Transform data 
# extract X and y from data, discarding UDIs, Product IDs and failure state reasons
X = data.iloc[:, 2:8].to_numpy()
y = data.iloc[:, 8].to_numpy().astype(np.float32)

# normalize features in X except for Type
from sklearn.preprocessing import StandardScaler
X[:,1:] = StandardScaler().fit_transform(X[:,1:])

# one hot encoding for Type feature
from sklearn.preprocessing import OneHotEncoder
X = np.hstack((OneHotEncoder().fit_transform(X[:,0].reshape(-1, 1)).toarray(), X[:,1:])).astype(np.float32)


# Picking the indices of the minority (failure) class
failure_indices = y.nonzero()[0].astype(np.int32)
# Picking the indices of the normal class
normal_indices = np.where(y == 0)[0].astype(np.int32)

number_failure = y.sum()

# Undersample and train/test models several times, with random undersampling
# to ensure that the undersampled datasets are representative of the whole dataset
NUM_LOOPS = 20

linreg_cnf_matrices = []
linreg_recall_scores = []
linreg_best_params = []
nn_cnf_matrices = []
nn_recall_scores = []
nn_best_params = []

for i in range(NUM_LOOPS):
    print("##### Iteration", i, "#####")

    # Out of the normal class indices, randomly select number_failure samples 
    random_normal_indices = np.random.choice(normal_indices, int(number_failure), replace = False)

    # Appending the indices of normal and failure classes
    under_sample_indices = np.concatenate([failure_indices, random_normal_indices]) 

    # Under sampled dataset
    X_undersample = X[under_sample_indices,:]
    y_undersample = y[under_sample_indices]
    
    # Undersampled dataset split
    X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = \
        train_test_split(X_undersample, y_undersample, test_size=0.3)
    

    ### Linear Regression training and testing

    #Apply function print_gridsearch_scores to get the best C with the Undersampled dataset
    #best_c = gridsearch_score(X_train_undersample, y_train_undersample)
    #linreg_best_params.append(best_c)

    # Use the best C to train LogReg model with undersampled train data and test it
    #lr = LogisticRegression(C = best_c)
    lr = LogisticRegression(C = 0.01, solver="liblinear")
    lr.fit(X_train_undersample,y_train_undersample)
    y_pred = lr.predict(X)
    cnf_matrix = confusion_matrix(y, y_pred)
    linreg_cnf_matrices.append(cnf_matrix)
    linreg_recall_scores.append(cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))


    ### NN training and testing

    # get best number and size of hidden layers for NN
    #best_params = gridsearch_score_deep_learning(X_train_undersample, y_train_undersample)
    #nn_best_params.append(best_params)

    # Use the best hidden_dimension to train and Deep NN model with the under-sample data and test it
    input_dim = X_train_undersample.shape[1]
    #k = KerasClassifier(build_fn=network_builder, epochs=100, batch_size=128,
    #                    hidden_dimensions=best_params, verbose=0, input_dim=input_dim)
    k = KerasClassifier(build_fn=network_builder, epochs=800, batch_size=128,
                        hidden_dimensions=[100, 100, 10], verbose=0, input_dim=input_dim)
    k.fit(X_train_undersample,y_train_undersample)
    y_pred = k.predict(X)
    cnf_matrix = confusion_matrix(y, y_pred)
    nn_cnf_matrices.append(cnf_matrix)
    nn_recall_scores.append(cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))


### calculate averages
tmp = np.dstack(linreg_cnf_matrices)
linreg_cnf_mean = tmp.mean(axis=2)
tmp = np.dstack(nn_cnf_matrices)
nn_cnf_mean = tmp.mean(axis=2)

### print results
with open("results.txt", "w") as fp:

    fp.write("---------- Linear Reg ----------\n")

    #for param, cnf, rec in zip(linreg_best_params, linreg_cnf_matrices, linreg_recall_scores):
    for cnf, rec in zip(linreg_cnf_matrices, linreg_recall_scores):
        #fp.write(f"C: {param}, Recall: {rec}\n")
        fp.write(f"C: {0.01}, Recall: {rec}\n")
        fp.write(str(cnf))
        fp.write("\n")
    
    fp.write("Mean confusion matrix:\n")
    fp.write(str(linreg_cnf_mean) + "\n")
    fp.write(f"Mean recall: {sum(linreg_recall_scores)/NUM_LOOPS}\n")
    
    fp.write("---------- NN ----------\n")

    #for param, cnf, rec in zip(nn_best_params, nn_cnf_matrices, nn_recall_scores):
    for cnf, rec in zip(nn_cnf_matrices, nn_recall_scores):
        #fp.write(f"Dims: {param}, Recall: {rec}\n")
        fp.write(f"Dims: {[100, 100, 10]}, Recall: {rec}\n")
        fp.write(str(cnf))
        fp.write("\n")
    
    fp.write("Mean confusion matrix:\n")
    fp.write(str(nn_cnf_mean) + "\n")
    fp.write(f"Mean recall: {sum(nn_recall_scores)/NUM_LOOPS}\n")
