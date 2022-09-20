from util import construct_data
from support_vector import svm_pegasos
from sklearn.datasets import make_blobs
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import svm


def svm_test():
    X, Y = make_blobs(n_samples=40, centers=2, cluster_std=1.2,
                      n_features=2, random_state=20)
    for i, j in enumerate(Y):
        if j == 0:
            Y[i] = -1
    df = pd.DataFrame(dict(x=X[:, 0], y=X[:, 1], label=Y))

    names = {-1: 'Group 1', 1: 'Group 2'}
    colors = {-1: (0, 100/255, 0, 0.9), 1: (138/255, 43/255, 226/255, 0.9)}
    grouped = df.groupby('label')

    x_test = X[20:]
    x_test = np.c_[x_test, np.ones(len(x_test))]
    y_test = Y[20:]
    # training sets
    x = X[:20]
    y = Y[:20]

    # group for plotting
    df_train = pd.DataFrame(dict(x=x[:, 0], y=x[:, 1], label=y))
    grouped_train = df_train.groupby('label')

    # add bias to sample vectors
    x = np.c_[x, np.ones(len(x))]

    custom_svm = svm_pegasos.SVM().fit(x, y)

    grid_res = 200

    # evenly split X range from min to max
    xline = np.linspace(min(X[:, 0]-(0.5*np.std(X[:, 0]))),
                        max(X[:, 0]+(0.5*np.std(X[:, 0]))), grid_res)

    # evenly split Y range from min to max
    yline = np.linspace(min(X[:, 1]-(0.5*np.std(X[:, 1]))),
                        max(X[:, 1]+(0.5*np.std(X[:, 1]))), grid_res)
    grid = []
    grid_color = []
    for i in range(grid_res):
        for j in range(grid_res):
            grid.append([xline[i], yline[j]])
            # if point is > 1 ->  categorize it as Group 2
            if (np.dot(custom_svm.w, [xline[i], yline[j], 1])) > 1:
                grid_color.append((138/255, 43/255, 226/255, 0.1))
            # if point < -1 -> categorize as Group 1
            elif (np.dot(custom_svm.w, [xline[i], yline[j], 1])) < -1:
                grid_color.append((0, 100/255, 0, 0.1))
            # if point == 0 -> it is on the decision boundry
            elif (round((np.dot(custom_svm.w, [xline[i], yline[j], 1])), 2) == 0):
                grid_color.append((0, 0, 0, 1))
            else:
                grid_color.append((245/255, 245/255, 245/255))

    grid = np.asarray(grid)
    grid_color = np.asarray(grid_color)
    # plot the data
    fig = plt.figure(figsize=(15, 11))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Feature 1", fontsize=18)
    ax.set_ylabel("Feature 2", fontsize=18)
    ax.scatter(grid[:, 0], grid[:, 1], marker='o', c=grid_color, s=10)
    for key, group in grouped_train:
        ax.scatter(group.x, group.y, label=names[key], color=colors[key], edgecolor=(
            0, 0, 0, 0.75), s=350)
    ax.legend(markerscale=1, fontsize=20, fancybox=True)
    plt.show()


def svm_ufc_test():
    X, y, X_test, y_test, X_future = construct_data()

    rows, columns = X.shape
    X = np.concatenate([np.ones((rows, 1)),
                        X], axis=1)
    X_future = np.concatenate([np.ones((X_future.shape[0], 1)),
                               X_future], axis=1)
    X_test = np.concatenate([np.ones((X_test.shape[0], 1)),
                             X_test], axis=1)

    print("------------SVM---------------")
    svm_clf = svm.SVC(C=5, gamma=0.01).fit(X, y)
    print("Score on test data")
    print(svm_clf.score(X_test, y_test))
    print("predictions for a future fight card")
    print(svm_clf.predict(X_future))
    print("number of support vectors")
    print(svm_clf.n_support_)
    print("------------Custom SVM---------------")
    zero_index = y == 0
    svm_y = y
    svm_y[zero_index] = -1
    custom_svm = svm_pegasos.SVM().fit(X, svm_y)
    print("predictions for a future fight card")
    print(custom_svm.predict_X(X_future))
    print("number of support vectors")
    print(custom_svm.positive_support)
    print(custom_svm.negative_support)


def svm_parameter_cross_val():
    # Custom method to determine kernel and c parameters

    X, y, X_test, y_test, X_future = construct_data()
    rows, columns = X.shape
    X = np.concatenate([np.ones((rows, 1)),
                        X], axis=1)
    X_future = np.concatenate([np.ones((X_future.shape[0], 1)),
                               X_future], axis=1)
    X_test = np.concatenate([np.ones((X_test.shape[0], 1)),
                             X_test], axis=1)

    kernel_list = ['linear', 'poly', 'rbf']
    c_array = np.linspace(0.1, 5, 49)
    high_score = 0
    best_c = None
    best_kernel = None
    for c in c_array:
        for kernel in kernel_list:
            svm_clf = svm.SVC(C=c, kernel=kernel).fit(X, y)
            score = svm_clf.score(X_test, y_test)
            print(f"Score for kernel {kernel} and C {c}:\n{score}\n")

            if score > high_score:
                best_c = c
                best_kernel = kernel
                high_score = score

    print(
        f"Best Kernel {best_kernel}  - Best C {best_c} - High score {high_score}")


def svm_grid_search():
    # A better way to determine hyperparameters using sklearn's GridSearchCV method
    # The result were: C = 5 and gamma=.01
    X, y, X_test, y_test, X_future = construct_data()
    c_array = np.linspace(1, 50, 50)
    print(c_array)

    param_grid = {'C': c_array, 'gamma': [
        1, 0.1, 0.01, 0.001], 'kernel': ['rbf', 'sigmoid']}
    grid = GridSearchCV(svm.SVC(), param_grid, refit=True, verbose=2)
    grid.fit(X, y)
    print(grid.best_estimator_)


def main():
    # svm_test()
    svm_ufc_test()
    # svm_parameter_cross_val()
   # svm_grid_search()


if __name__ == "__main__":
    main()
