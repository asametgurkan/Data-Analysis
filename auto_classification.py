import numpy as np
import matplotlib.pyplot as plt
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
        self.X = X
        self.y = y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.33)

    def predict_model(self, model_name):
        model_name = model_name.lower()
        if model_name == 'decisiontree':
            model = DecisionTreeClassifier()
        elif model_name == 'svm':
            model = SVC()
        elif model_name == 'randomforest':
            model = RandomForestClassifier()
        elif model_name == 'knn':
            model = KNeighborsClassifier()
        elif model_name == 'logisticregression':
            model = LogisticRegression()
        elif model_name == 'gaussiannb':
            model = GaussianNB()
        elif model_name == 'multinomialnb':
            model = MultinomialNB()
        elif model_name == 'bernoullinb':
            model = BernoulliNB()
        else:
            raise ValueError("Invalid model name.")
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        scores_f1 = cross_val_score(model, self.X, self.y, cv=10, scoring='f1_macro')
        scores_acc = cross_val_score(model, self.X, self.y, cv=10, scoring='accuracy')
        return {
            "f1_score": f1_score(predictions, self.y_test, average="macro").round(2),
            "accuracy_score": accuracy_score(predictions, self.y_test).round(2),
            "f1_score_cross_val_mean": scores_f1.mean().round(2),
            "accuracy_score_cross_val_mean": scores_acc.mean().round(2)
        }

    def collect_model_scores(self):
        model_scores = {}
        model_names = ['DecisionTree', 'SVM', 'RandomForest', 'KNN', 'LogisticRegression', 'GaussianNB', 'MultinomialNB', 'BernoulliNB']
        for model_name in model_names:
            model_scores[model_name] = self.predict_model(model_name)
        return model_scores
    def all_model_performance(self):
        all_model_scores = model_classifier.collect_model_scores()
        f1_scores_list=[]
        acc_scores_list=[]
        model_names = ['DecisionTree', 'SVM', 'RandomForest', 'KNN', 'LogisticRegression', 'GaussianNB', 'MultinomialNB', 'BernoulliNB']
        for model_name in all_model_scores.keys():
            f1_scores_list.append(all_model_scores[model_name]['f1_score_cross_val_mean'])
            acc_scores_list.append(all_model_scores[model_name]['accuracy_score_cross_val_mean'])
        df_f1=pd.DataFrame({"Model Names":model_names,"F1 Scores":f1_scores_list})
        df_acc=pd.DataFrame({"Model Names":model_names,"Accuracy Scores":acc_scores_list})
        merged_df = pd.merge(df_f1, df_acc, on="Model Names")
        plot = merged_df.plot(x="Model Names", y=["F1 Scores", "Accuracy Scores"], kind="bar")
        plot.set_xlabel("Model Names")
        plot.set_ylabel("Scores")
        plt.title("F1 Scores and Accuracy Scores for Different Models")
        plt.show()

model_classifier = ClassificationModels(X, y)
model_input = input("Enter the model name (DecisionTree, RandomForest, etc.) : ")
model_classifier.predict_model(model_input)


    
