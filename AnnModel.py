# Train ANN
import keras
from ANN.prepration_data import clean_dataframe
from sklearn.externals import joblib
from sklearn import preprocessing
from keras.models import load_model
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def Ann_Model(train_data):

    X = train_data[["Tn-1", "Tn-2", 'Tn-3', 'Tn-4', 'Tn-5',
                    'Tn-6', 'Tn-7', 'Tn-8', 'Tn-9', 'Tn-10',
                    'ID', 'tests_total_sum', 'prob_skipped']]

    X[["Tn-1", "Tn-2", 'Tn-3', 'Tn-4', 'Tn-5', 'Tn-6', 'Tn-7', 'Tn-8', 'Tn-9', 'Tn-10', 'tests_total_sum']] = X[["Tn-1", "Tn-2", 'Tn-3', 'Tn-4', 'Tn-5', 'Tn-6', 'Tn-7', 'Tn-8', 'Tn-9', 'Tn-10', 'tests_total_sum']]/100

    # Clustering
    clustering_model = KMeans(n_clusters=10, random_state=0).fit(X)
    X['Cluster_Label'] = "No label"
    X['Cluster_Label'] = clustering_model.labels_

    X = X[["Tn-1", "Tn-2", 'Tn-3', 'Tn-4', 'Tn-5', 'Tn-6', 'Tn-7', 'Tn-8', 'Tn-9', 'Tn-10', 'tests_total_sum', 'prob_skipped','Cluster_Label']]
    # Save the model
    clustering = 'C:/Dev/...'
    joblib.dump(clustering_model, clustering)
    Y = train_data['Tn']
    print(X)

    # feature scaling Y
    scalary = preprocessing.MinMaxScaler()
    y = (np.array(Y)).reshape(-1, 1)
    y = scalary.fit_transform(y)
    scalar_y = "C:/Dev/..."
    joblib.dump(scalary, scalar_y)
    X, X_val, y, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

    # ##############################################
    #
    # #  define a regression model
    # def regression_model():
    #     # create model
    #     model = Sequential()
    #     model.add(Dense(15, input_dim=13, kernel_initializer='random_uniform', activation='tanh'))
    #     model.add(Dense(30, kernel_initializer='random_uniform', activation='tanh'))
    #     model.add(Dense(30, kernel_initializer='random_uniform', activation='tanh'))
    #     model.add(Dense(1, kernel_initializer='random_uniform', activation='linear'))
    #     # Compile model
    #     ADAM = keras.optimizers.Adam(lr=0.00001, beta_1=0.98, beta_2=0.999, decay=0)
    #     model.compile(loss='mean_squared_error', optimizer=ADAM, metrics=['mse', 'mae'])
    #     return model
    # ##################################################

    # model = regression_model()
    # model.save('C:/Dev/campaign-optimization/data/model.h5')
    model = load_model('C:/Dev/...')
    es = keras.callbacks.EarlyStopping(monitor='val_loos', patience=10, verbose=0)
    history = model.fit(X, y, validation_split=0.33, epochs=300, verbose=1, callbacks=[es])
    y_predict = model.predict(X)
    model.save('C:/Dev/model.h5')
    # load best score of the model
    test_acc = model.evaluate(X_val, y_val, verbose=0)
    last_acc = np.load('C:/Dev/best-acc.npy')
    print("the accuracy of the validation set is :")
    print(test_acc)

    # update weight of model when we have a better accuracy
    if last_acc > test_acc[1]:
        print(last_acc)
        model.save('C:/Dev/model.h5')
        last_score = test_acc[1]
        np.save('C:/Dev/best-acc', last_score)

    plt.figure(1)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Test')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')

    plt.figure(2)
    plt.title('Mean Squared Error')
    plt.plot(history.history['mean_squared_error'], label='train')
    plt.plot(history.history['val_mean_squared_error'], label='test')
    plt.legend(['train', 'validation'], loc='upper left')

    fig, ax = plt.subplots()
    ax.scatter(y, y_predict)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')

    plt.show()

    return 0


if __name__ == '__main__':

    # result = clean_dataframe(8)
    result = pd.read_csv('C:/Dev/Train.csv')

    history = Ann_Model(result)
    print(history)


