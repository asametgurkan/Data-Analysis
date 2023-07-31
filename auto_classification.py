from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

class ClassificationModels:
    def __init__(self, X, y):
        self.X=X
        self.y=y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    print("************\nChoose model: \nDecisionTree\nSVM\nRandomForest\nKNN\nLogisticRegression\nGaussianNB\nMultinomialNB\nBernoulliNB\n************")

    def predict_model(self, model_name):
        if model_name == 'DecisionTree':
            model = DecisionTreeClassifier()
        elif model_name == 'SVM':
            model = SVC()
        elif model_name == 'RandomForest':
            model = RandomForestClassifier()
        elif model_name == 'KNN':
            model = KNeighborsClassifier()
        elif model_name == 'LogisticRegression':
            model = LogisticRegression()
        elif model_name == 'GaussianNB':
            model = GaussianNB()
        elif model_name == 'MultinomialNB':
            model = MultinomialNB()
        elif model_name == 'BernoulliNB':
            model = BernoulliNB()
        else:
            raise ValueError("Invalid model name.")
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        scores_f1 = cross_val_score(model, self.X, self.y, cv=10, scoring='f1_macro')
        scores_acc = cross_val_score(model, self.X, self.y, cv=10, scoring='accuracy')
        print("Model:", model_name)
        print("f1 Score:", f1_score(predictions, self.y_test, average="macro").round(2))
        print("Accuracy Score:", accuracy_score(predictions, self.y_test).round(2))
        print("f1 score cross-validated mean:", scores_f1.mean().round(2))
        print("Accuracy cross-validated mean:", scores_acc.mean().round(2))
model_classifier = ClassificationModels(X, y)
model_classifier.predict_model("SVM")

