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
    svm_clf = svm.SVC(C=1).fit(X, y)
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
