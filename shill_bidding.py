from operator import indexOf
from six import StringIO
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import model_selection, metrics, tree, linear_model, feature_selection, neighbors, cluster
import seaborn as sns

#read csv file
def read_csv():
    df=pd.read_csv('Shill Bidding Dataset.csv')
    return df

#rows and columns
def rows_columns(df):
    rows=len(df)
    columns=len(df.columns)
    print("-----Row number: {}\n-----Column number: {}".format(rows, columns))
    return[rows, columns]

#remove unnecessary features
def remove_unnecessary(df):
    del df["Record_ID"]
    del df["Auction_ID"]
    del df["Bidder_ID"]
    print("-----Data after deleting unnecessary features:")
    print(df.head(10))
    return df

#split data on test and train samples
def shaffle_split(df, N):
    var_columns=[i for i in df.columns if i not in ['Class']]
    X=df.loc[:, var_columns]
    Y=df.loc[:, "Class"]
    ss=model_selection.ShuffleSplit(n_splits=N, test_size=.2, random_state=0)
    for train_index, test_index in ss.split(X, Y):
        x=1

    train=df.iloc[train_index]
    test=df.iloc[test_index]
    return train, test

def xsys(train, test, var_columns):
    x_train=train.loc[:, var_columns]
    y_train=train.loc[:, "Class"]
    x_test=test.loc[:, var_columns]
    y_test=test.loc[:, "Class"]

    return x_train, y_train, x_test, y_test

#build a decision tree model
def decision_tree(df, N):

    train=shaffle_split(df, N)[0]
    test=shaffle_split(df, N)[1]

    var_columns=[i for i in train.columns if i not in ['Class']]

    x_train, y_train, x_test, y_test=xsys(train, test, var_columns)

    tree_model=tree.DecisionTreeClassifier(max_depth=5, class_weight="balanced")
    tree_model=tree_model.fit(x_train, y_train)

    return tree_model, var_columns, x_train, x_test, y_train, y_test

#draw decision tree
def draw_tree(tree_model, var_columns_train):
    dot_data=StringIO()
    tree.export_graphviz(tree_model, out_file="graph.dot", feature_names=var_columns_train, class_names="Class", filled=True, rounded=True, special_characters=True)

#classification metrics
def class_metrics(x_test, y_test, x_train, y_train, model) :
    
    y_pred_train=model.predict(x_train)
    y_pred_test=model.predict(x_test)

    print("-----Test classification report:\n",metrics.classification_report(y_test, y_pred_test))
    print("-----Train classification report:\n",metrics.classification_report(y_train, y_pred_train))

    cm=metrics.confusion_matrix(y_train, y_pred_train)
    plt.figure(figsize=(5,5))
    sns.heatmap(data=cm,linewidths=.5, annot=True,square = True,  cmap = 'Blues')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Score of test data: {0:.2f}'.format(model.score(x_test, y_test))
    plt.title(all_sample_title, size = 15)
    plt.show()
    return model.score(x_train, y_train), model.score(x_test, y_test)

    

#decision tree metrics
def dt_metrics(x_test, y_test, x_train, y_train):
    t_m = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)
    t_m=t_m.fit(x_train, y_train)
    y_pred_test_1=t_m.predict(x_test)
    entropy_accuracy=metrics.accuracy_score(y_test, y_pred_test_1)
    print("-----Accuracy entropy:",entropy_accuracy)

    t_m = tree.DecisionTreeClassifier(criterion="gini", max_depth=5)
    t_m=t_m.fit(x_train, y_train)
    y_pred_test_1=t_m.predict(x_test)
    gini_accuracy=metrics.accuracy_score(y_test, y_pred_test_1)
    print("-----Accuracy Gini:",gini_accuracy)


    if entropy_accuracy>gini_accuracy:
        print("Accuracy is higher when applying the entropy splitting criterion")
    else:
        print("Accuracy is higher when applying the Gini splitting criterion")


#investigate the inpact of the min samples split on decision tree prediction
def min_samples_split_impact(x_train, y_train, x_test, y_test):
    min_samples_splits=np.linspace(0.1, 1.0, 10, endpoint=True)

    train_results=[]
    test_results=[]
    for min_samples_split in min_samples_splits:
        dt=tree.DecisionTreeClassifier(min_samples_split=min_samples_split)
        dt.fit(x_train, y_train)

        y_predict_train=dt.predict(x_train)
        false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_train, y_predict_train)
        roc_auc = metrics.auc(false_positive_rate, true_positive_rate)
        train_results.append(roc_auc)

        y_predict_test=dt.predict(x_test)

        false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_test, y_predict_test)
        roc_auc = metrics.auc(false_positive_rate, true_positive_rate)
        test_results.append(roc_auc)

    line1, = plt.plot(min_samples_splits, train_results,color="blue", label="Train AUC")
    line2, = plt.plot(min_samples_splits, test_results, color="red", label="Test AUC")

    plt.title("Min samples split impact")
    plt.legend()
    plt.ylabel("AUC score")
    plt.xlabel("min samples split")
    plt.show()

#investigate the inpact of the max leaf nodes on decision tree prediction
def max_leaf_impact(x_train, y_train, x_test, y_test):
    max_leaves=np.arange(2, 11, 2)

    train_results=[]
    test_results=[]
    for max_leaf in max_leaves:
        dt=tree.DecisionTreeClassifier(max_leaf_nodes=max_leaf)
        dt.fit(x_train, y_train)

        y_predict_train=dt.predict(x_train)

        false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_train, y_predict_train)
        roc_auc = metrics.auc(false_positive_rate, true_positive_rate)
        train_results.append(roc_auc)

        y_predict_test=dt.predict(x_test)

        false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_test, y_predict_test)
        roc_auc = metrics.auc(false_positive_rate, true_positive_rate)
        test_results.append(roc_auc)

    line1, = plt.plot(max_leaves, train_results,color="blue", label="Train AUC")
    line2, = plt.plot(max_leaves, test_results, color="red", label="Test AUC")

    plt.title("Max Leaf nodes impact")
    plt.legend()
    plt.ylabel("AUC score")
    plt.xlabel("max leaves")
    plt.show()

#calculate importance of the features
def feature_importances(tree_model, var_columns_train):
    feature_importance=tree_model.feature_importances_
    print("-----Feature importances:\n", feature_importance)
    print("-----The importance of a feature is computed as the normalized total reduction of the criterion brought by that feature.\nN_t / N * (impurity - N_t_R / N_t * right_impurity- N_t_L / N_t * left_impurity),     where N is the total number of samples, N_t is the number of samples at the current node, N_t_L is the number of samples in the left child, and N_t_R is the number of samples in the right child.")
    plt.bar(var_columns_train,feature_importance)
    plt.show()


#build Logistic Regression model
def log_reg(x_train, y_train):
    lr=linear_model.LogisticRegression()
    lr=lr.fit(x_train, y_train)
    return lr

#invesigate the impact of the regularization parametr 
def regularization_impact(x_train, y_train, x_test, y_test):
    regularization_parameter =np.arange(1, 10001, 1000)

    train_results=[]
    test_results=[]
    for regularization in regularization_parameter:
        lr=linear_model.LogisticRegression(C=regularization)
        lr.fit(x_train, y_train)

        y_predict_train=lr.predict(x_train)

        false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_train, y_predict_train)
        roc_auc = metrics.auc(false_positive_rate, true_positive_rate)
        train_results.append(roc_auc)

        y_predict_test=lr.predict(x_test)

        false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_test, y_predict_test)
        roc_auc = metrics.auc(false_positive_rate, true_positive_rate)
        test_results.append(roc_auc) 

    line1, = plt.plot(regularization_parameter, train_results,color="blue", label="Train AUC")
    line2, = plt.plot(regularization_parameter, test_results, color="red", label="Test AUC")

    plt.title("Regularization parameter impact")
    plt.legend()
    plt.ylabel("AUC score")
    plt.xlabel("regularization parameter")
    plt.show()

#calculate importance of the features
def importance(lr, var_columns, x_train, y_train, train, test):
    selector = feature_selection.RFE(lr, n_features_to_select=1, step=1)
    selector = selector.fit(x_train, y_train)
    ranking=selector.ranking_
    rank_dict={}
    for i in range(len(var_columns)):
        rank_dict[var_columns[i]]=ranking[i]
    rank_dict=dict(sorted(rank_dict.items(), key=lambda item: item[1]))
    print("-----Features ranking: {}".format(rank_dict))
    selector = feature_selection.RFE(lr, n_features_to_select=5, step=1)
    selector = selector.fit(x_train, y_train)
    ranking=selector.ranking_
    important_features=[]
    for i in range(len(ranking)):
        if ranking[i]==1:
            important_features.append(var_columns[i])
    print("-----The most important features: {}".format(important_features))
    x_train_new=xsys(train, test, important_features)[0]
    y_train_new=xsys(train, test, important_features)[1]
    x_test_new=xsys(train, test, important_features)[2]
    y_test_new=xsys(train, test, important_features)[3]
    lr=log_reg(x_train_new, y_train_new )
    y_pred_train=lr.predict(x_train_new)
    y_pred_test=lr.predict(x_test_new)
    print("-----Test classification report:\n",metrics.classification_report(y_test_new, y_pred_test))
    print("-----Train classification report:\n",metrics.classification_report(y_train_new, y_pred_train))

    cm=metrics.confusion_matrix(y_test_new, y_pred_test)
    plt.figure(figsize=(5,5))
    sns.heatmap(data=cm,linewidths=.5, annot=True,square = True,  cmap = 'Blues')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Score of test data: {0:.2f}'.format(lr.score(x_test_new, y_test_new))
    plt.title(all_sample_title, size = 15)
    plt.show()


#build K-Neighbors model
def k_neighbors(x_train, y_train):
    kn=neighbors.KNeighborsClassifier()
    kn=kn.fit(x_train, y_train)
    return kn

#investigate the impact of the leaf size parametr
def kdtree_impact(x_train, y_train, x_test, y_test):
    kdtree_parameter =np.arange(20, 81, 2)

    train_results=[]
    test_results=[]
    for kdtree in kdtree_parameter:
        kn=neighbors.KNeighborsClassifier(algorithm="kd_tree", leaf_size=kdtree)
        kn.fit(x_train, y_train)

        y_predict_train=kn.predict(x_train)

        false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_train, y_predict_train)
        roc_auc = metrics.auc(false_positive_rate, true_positive_rate)
        train_results.append(roc_auc)

        y_predict_test=kn.predict(x_test)

        false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_test, y_predict_test)
        roc_auc = metrics.auc(false_positive_rate, true_positive_rate)
        test_results.append(roc_auc) 

    line1, = plt.plot(kdtree_parameter, train_results,color="blue", label="Train AUC")
    line2, = plt.plot(kdtree_parameter, test_results, color="red", label="Test AUC")

    plt.title("Leaf size impact")
    plt.legend()
    plt.ylabel("AUC score")
    plt.xlabel("leaf size")
    plt.show()


#Set new columns of indexes
def set_idx(df):
    # print("TASK 3")
    df=df.set_index("Record_ID")
    return df

#investigate the impact of number of clusters
def n_cluster_opt(df, method):
    
    del df["Bidder_ID"]
    range_n_clasters=[2,3,4,5,6,7,8,9]
    silhouette_avg=[]
    if method=="KMeans":
        for n_cluster in range_n_clasters:
            kmeans=cluster.KMeans(n_clusters=n_cluster)
            kmeans.fit(df)
            cluster_lables=kmeans.labels_
            silhouette_avg.append(metrics.silhouette_score(df, cluster_lables))

        n_cluster=range_n_clasters[silhouette_avg.index(max(silhouette_avg))]
        plt.plot(range_n_clasters,silhouette_avg,'bx-')
        plt.xlabel("Values of K")
        plt.ylabel("Silhouette score")
        plt.title("Silhouette analysis For Optimal k")
        plt.show()
    else:
        for n_cluster in range_n_clasters:
            kmeans=cluster.AgglomerativeClustering(n_clusters=n_cluster)
            kmeans.fit(df)
            cluster_lables=kmeans.labels_
            silhouette_avg.append(metrics.silhouette_score(df, cluster_lables))

        n_cluster=range_n_clasters[silhouette_avg.index(max(silhouette_avg))]
        plt.plot(range_n_clasters,silhouette_avg,'bx-')
        plt.xlabel("Values of K")
        plt.ylabel("Silhouette score")
        plt.title("Silhouette analysis For Optimal k")
        plt.show()

    return n_cluster

#Clustering using K Mean method
def kmeans(df):
    n_cluster=n_cluster_opt(read_csv(), "KMeans")
    kmeans=cluster.KMeans(n_clusters=n_cluster)
    kmeans.fit(df)
    print("Clusters centers:")
    print(kmeans.cluster_centers_)
    cluster_lables=kmeans.labels_
    score=metrics.silhouette_score(df, cluster_lables)
    print("The accuracy using K Means function equals {}".format(score))
    return score

#clustering using Agglomerative Clustering method
def agglomerative(df):
    n_cluster=n_cluster_opt(read_csv(), "AgglomerativeClustering")
    agg=cluster.AgglomerativeClustering(n_clusters=n_cluster).fit(df)
    clust=cluster.AgglomerativeClustering(n_clusters=n_cluster).fit_predict(df)
    clf=neighbors.NearestCentroid()
    clf.fit(df, clust)
    print(clf.centroids_) 

    cluster_lables=agg.labels_
    score=metrics.silhouette_score(df, cluster_lables)
    
    print("The accuracy using AgglomerativeClustering function equals{}".format(score))
    return score

#chose method with higher accuracy
def choose(kmean_score, agg_score):
    if kmean_score>agg_score:
        print("It's beeter to use KMean function, because the accuracy is higher")
        return kmean_score
    else:
        print("It's beeter to use AgglomerativeClustering, because the accuracy is higher")
        return agg_score


#main function
def run():
    #Define varaible to compare method effectivness later
    score={"Decision Tree":[], "Logistic Regression":[], "K-Neighbours":[], "Clustering":[]}

    #Remove unnecessary features from data
    df=remove_unnecessary(read_csv())

    #choose split variant
    N=int(input("Choose split variant (input number less than 20): "))

    ##Decision Tree
    #build Decision Tree model and split data into train and test samples
    tree_model=decision_tree(df, N)[0]
    train=shaffle_split(df, N)[0]
    test=shaffle_split(df, N)[1]
    var_columns=[i for i in train.columns if i not in ['Class']]
    x_train=decision_tree(df, N)[2]
    x_test=decision_tree(df, N)[3] 
    y_train=decision_tree(df, N)[4]
    y_test=decision_tree(df, N)[5]
    print("-----{}th iteration train split: \n{}\n-----{}th iteration test split: \n{}".format(N, train, N, test))

    #draw Decission Tree
    draw_tree(tree_model, var_columns)

    #Calculate classification metrix and accuracy of the model
    score["Decision Tree"]=list(class_metrics(x_test, y_test, x_train, y_train, tree_model))
    dt_metrics(x_test, y_test, x_train, y_train)

    #investigate impact of factors on prediction of the model
    min_samples_split_impact(x_train, y_train, x_test, y_test)
    max_leaf_impact(x_train, y_train, x_test, y_test)
    feature_importances(tree_model, var_columns)

    ##Logistic Regression
    #build Logistic Regression model
    lr=log_reg(x_train, y_train)
    score["Logistic Regression"]=list(class_metrics(x_train, y_train, x_test, y_test, lr))

    #investigate impact of factors on prediction of the model
    regularization_impact(x_train, y_train, x_test, y_test)
    importance(lr, var_columns, x_train, y_train, train, test)

    ##K-Neighbors
    #build K-Neighbors model
    kn=k_neighbors(x_train, y_train)
    score["K-Neighbours"]=list(class_metrics(x_train, y_train, x_test, y_test, kn))

    #investigate impact of factors on prediction of the model
    kdtree_impact(x_train, y_train, x_test, y_test)

    ##Clustering
    #clean data
    df=set_idx(read_csv())
    del df["Auction_ID"]
    del df["Bidder_ID"]
    del df["Class"]
    print("-----Data after deleting unnecessary features:")
    print(df.head(10))
    
    #clustering using different methods
    kmean_score=kmeans(df)
    agg_score=agglomerative(df)
    score["Clustering"]=[0, choose(kmean_score, agg_score)]

    ##Best model
    train_score, test_score=[], []
    for key in score.keys():
        train_score.append(score[key][0])
        test_score.append(score[key][1])

    #plot scores of each model
    plt.bar(score.keys(), test_score, color='red', label="Test data accuracy")
    plt.stem(score.keys(), train_score, label="Train data accuracy")
    plt.legend()
    plt.xlabel("Method names")
    plt.ylabel("Accuracy score")
    plt.show()

    #choose the most accurate model
    best_method=list(score.keys())[test_score.index(max(test_score))]
    print(f"-----{best_method} method gives the most accurate result")



run()
