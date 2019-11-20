import numpy as np
from preprocess import data
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

def model(dataObj):
    X = dataObj.numerical
    Y = dataObj.stars
    logreg = LogisticRegression(penalty = "l2", C = 1e5, solver = "lbfgs",\
        max_iter = 200, multi_class = "multinomial")
    scores = cross_val_score(logreg, X, Y, cv = 10)
    return scores

if __name__ == "__main__":
    dataObj = data("Yelp_train.csv")
    dataObj.center()
    scores = model(dataObj)
    print("Cross validation scores:")
    print(scores)
