# Load libraries
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
from libmlops.models.model_evaluation import cross_validate_model
from sklearn.metrics import mean_absolute_error, r2_score

# Spot Check Algorithms
models = [
    ('LINR', LinearRegression(n_jobs=4)),
    ('RDG', Ridge()),
    ('LSO', Lasso()),
    ('DTR', DecisionTreeRegressor()),
    ('RFR', RandomForestRegressor(n_jobs=4)),
    ('SVR', SVR(kernel='linear')),
    ('KNR', KNeighborsRegressor(n_jobs=4)),
    ('GBR', GradientBoostingRegressor()),
    # ('LOGR', LogisticRegression(n_jobs=4)),
]

def regressor_evaluation(X_train, Y_train, verbose=False):
    # evaluate each model in turn
    results = []
    names = []
    for name, model in models:
        kfold = KFold(n_splits=10, random_state=1, shuffle=True)
        cv_results_mean, cv_results_std = cross_validate_model(model, X_train, Y_train, cv=kfold, scoring='neg_mean_absolute_error')
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
