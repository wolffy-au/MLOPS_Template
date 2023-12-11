# Load libraries
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from models.model_evaluation import cross_validate_model

# Spot Check Algorithms
models = [
    ('LR', LogisticRegression(solver='liblinear', multi_class='ovr')),
    ('LDA', LinearDiscriminantAnalysis()),
    ('KNN', KNeighborsClassifier()),
    ('CART', DecisionTreeClassifier()),
    ('NB', GaussianNB()),
    ('SVM', SVC(gamma='auto'))
]

def algorithm_evaluation(X_train, Y_train, verbose=False):
    # evaluate each model in turn
    results = []
    names = []
    for name, model in models:
        kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
        cv_results_mean, cv_results_std = cross_validate_model(model, X_train, Y_train, cv=kfold, scoring='accuracy')
        results.append([cv_results_mean, cv_results_std])
        names.append(name)
        if verbose:
            print('%s: %f (%f)' % (name, cv_results_mean, cv_results_std))
    return results, names

def compare_algorithms(results, names):
    # Compare Algorithms
    plt.boxplot(results, labels=names)
    plt.title('Algorithm Comparison')
    plt.show()
