from sklearn import tree


def classify(features_train, labels_train, min_split):
    ### your code goes here--should return a trained decision tree classifer

    clf = tree.DecisionTreeClassifier(min_samples_split=min_split)
    clf = clf.fit(features_train, labels_train)

    return clf