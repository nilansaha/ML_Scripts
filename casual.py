import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

class LinearFeatureSelection:
    '''
    Select Features using a feature weights from linear models
    '''
    def __init__(self, classification, threshold = 0.15):
        self.classification = classification
        self.threshold = threshold
        self.relevant_indices = None
    
    def fit(self, X, y):
        if len(np.unique(y)) != 2:
            raise ValueError('Only Supported for binary problems')

        if self.classification:
            lr = LogisticRegression()
        else:
            lr = LinearRegression()
        
        lr.fit(X, y)
        normalized = abs(lr.coef_[0]/np.max(abs(lr.coef_[0])))

        self.relevant_indices = np.where(normalized > self.threshold)[0]
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.relevant_indices]
        else:
            return X[:, self.relevant_indices]

class FastFeatureImportance:
    '''
    Finding feature importance in O(n) by adding random data to features one at a time
    '''
    def __init__(self, estimator):
        self.estimator = estimator
        self.importance = []

    def fit(self, X, y):
        features = list(range(X.shape[1]))
        length = len(X)
        self.estimator.fit(X, y)
        base_score = self.estimator.score(X, y)
        if isinstance(X, pd.DataFrame):
            X = X.values
        for feature in features:
            temp = X
            temp[:, feature] = np.random.normal(0, 0.1, length)
            self.estimator.fit(temp, y)
            importance = round(abs(base_score - self.estimator.score(temp, y)), 5)
            self.importance.append(importance)

    @property
    def importance_(self):
        return self.importance
