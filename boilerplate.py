import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


df = pd.read_csv('train.csv')

X = df.drop(['Output'], axis = 1)
y = df['Output']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.30, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_valid, y_valid, test_size = 0.50, random_state=42)

numeric_features = ['Feature_A', 'Feature_B']
categorical_features = ['Feature_C', 'Feature_D']

preprocessor = ColumnTransformer(
    transformers = [
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown = 'ignore'), categorical_features)
    ]
)

X_train = pd.DataFrame(preprocessor.fit_transform(X_train).toarray(),
                       index=X_train.index,
                       columns=(numeric_features +
                                list(preprocessor.named_transformers_['cat']
                                     .get_feature_names(categorical_features))))

X_valid = pd.DataFrame(preprocessor.transform(X_valid).toarray(),
                       index=X_valid.index,
                       columns=(numeric_features +
                                list(preprocessor.named_transformers_['cat']
                                     .get_feature_names(categorical_features))))


dummy_clf = DummyClassifier(random_state = 42)
dummy_clf.fit(X_train, y_train)
dummy_predictions = dummy_clf.predict(X_valid)
print('Baseline Score', f1_score(y_valid, dummy_predictions, average = 'macro'))

