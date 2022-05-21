import math
import pandas as pd
import numpy as np
from sklearn import metrics

#Split dataframe by class
def class_dict(dataset):
    separated = dict()
    classes = df.iloc[:, -1].unique()
    for class_value in classes:
        if class_value not in separated:
            separated[class_value] = df[df[df.columns[-1]] == class_value]
    return separated
 
#Mean and standard deviation of each class
def class_mean_std(df):
    separated = class_dict(df)
    summaries = dict()
    for class_value, data in separated.items():
        summaries[class_value] = [(data[column].mean(), data[column].std(), len(data[column])) for column in data.columns[:-1]]
    return summaries
 
#Get conditional probability (P(X|Class)) using gaussian distribution 
def conditional_probability(x, mean, stddev):
    return (1 / (math.sqrt(2 * math.pi) * stddev)) * math.exp(-(((x - mean) ** 2) / (2 * (stddev ** 2))))
 
#Probability of each class for given prediction vector
def predict_value(summaries, prediction_vector):
    #Total length of dataset from class wise split
    total_length = sum([summaries[class_value][0][2] for class_value in summaries])
    probabilities = dict()
    for class_value, class_summaries in summaries.items():
        #P[class] = len(class_split) / total_length_of_dataset 
        probabilities[class_value] = summaries[class_value][0][2]/float(total_length)
        for i in range(len(class_summaries)):
            mean, stddev, class_length = class_summaries[i]
            #P[class|X1, ... , Xn] = (P[X|Class] * ... * P[Xn|Class]) * P[Class] {P[Class] is already calculated above}
            probabilities[class_value] *= conditional_probability(prediction_vector[i], mean, stddev)
    return probabilities

def predict(dataset):
    summaries = class_mean_std(df)
    Y_pred = list()
    for i in range(len(dataset)):
        probabilities = predict_value(summaries, list(df.iloc[i, :]))
        class_prediction = np.argmax(list(probabilities.values()))
        Y_pred.append(class_prediction)
    return Y_pred
 
if __name__ == "__main__":
    df = pd.read_csv("iris.csv")
    Y_true = df["class"].to_numpy()
    Y_pred = np.array(predict(df))
    print(df)
    print("Accuracy : ", metrics.accuracy_score(Y_true, Y_pred))
    print("Precision : ", metrics.precision_score(Y_true, Y_pred, average = "micro"))
    print("Confusion Matrix : \n", metrics.confusion_matrix(Y_true, Y_pred))
