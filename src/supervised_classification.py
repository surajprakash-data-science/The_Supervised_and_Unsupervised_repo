import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

if __name__ == "__main__":
    from data_ingestion import load_data, preprocess_data

    df = load_data("")
    df = preprocess_data(df)
    le = LabelEncoder()
    df['target'] = le.fit_transform(df['target'])  # Example of encoding a

    variable_coloumn = df[["feature1", "feature2"]]  
    target_column = df["target"]  

    X_train, X_test, y_train, y_test = train_test_split(variable_coloumn, target_column, test_size=0.2, random_state=42)

    # Train a simple logistic regression model
    log_model = LogisticRegression(
        solver='liblinear', # Suitable for small datasets and binary classification
        penalty='l2',       # L2 regularization
        C=1.0,              # Inverse of regularization strength; smaller values specify stronger regularization
        random_state=42     # For reproducibility
    )
    log_model.fit(X_train, y_train)

    y_pred = log_model.predict(X_test)

    # Evaluate the model
    print("#############################################################################")
    print("Logistic Regression Model Evaluation:")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("5 values of original vs predicted:")
    print("Actual:", y_test.head().values, "\nPredicted:", y_pred[:5])


    # Train a simple decision tree classifier
    tree_model = DecisionTreeClassifier(
        random_state=42,
        max_depth=3,          # Limit the depth of the tree to prevent overfitting
        min_samples_split=4,   # Minimum samples required to split an internal node
        min_samples_leaf=2    # Minimum samples required to be at a leaf node
    )
    tree_model.fit(X_train, y_train)

    y_pred_tree = tree_model.predict(X_test)    

    # Evaluate the decision tree model
    print("#############################################################################")
    print("Decision Tree Classifier Model Evaluation:")
    print("Classification Report:\n", classification_report(y_test, y_pred_tree))
    print("Accuracy:", accuracy_score(y_test, y_pred_tree))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_tree))
    print("5 values of original vs predicted:")
    print("Actual:", y_test.head().values, "\nPredicted:", y_pred_tree[:5])


    # Train a simple random forest classifier
    forest_model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        max_depth=4, 
        min_samples_split=4, 
        min_samples_leaf=2, 
        n_jobs=-1)
    forest_model.fit(X_train, y_train)

    y_pred_forest = forest_model.predict(X_test)

    # Evaluate the random forest model
    print("#############################################################################")
    print("Random Forest Classifier Model Evaluation:")
    print("Classification Report:\n", classification_report(y_test, y_pred_forest))
    print("Accuracy:", accuracy_score(y_test, y_pred_forest))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_forest))
    print("5 values of original vs predicted:")
    print("Actual:", y_test.head().values, "\nPredicted:", y_pred_forest[:5])



    # Train a SVM classifier model
    #======================================================================
    svm_model = SVC(
        C=1.0,                    # Regularization parameter
        kernel='rbf',             # Radial basis function kernel
        gamma='scale',            # Kernel coefficient
        probability=True,         # Enable probability estimates
        random_state=42
    )
    svm_model.fit(X_train, y_train)

    y_pred_svm = svm_model.predict(X_test)

    # Evaluate the SVM model
    print("#############################################################################")
    print("SVM Classifier Model Evaluation:")
    print("Classification Report:\n", classification_report(y_test, y_pred_svm))
    print("Accuracy:", accuracy_score(y_test, y_pred_svm))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))
    print("5 values of original vs predicted:")
    print("Actual:", y_test.head().values, "\nPredicted:", y_pred_svm[:5])


    # Train a K-Nearest Neighbors classifier model
    #======================================================================
    knn_model = KNeighborsClassifier(
        n_neighbors=5,            # Number of neighbors to use
        weights='uniform',        # All points in each neighborhood are weighted equally
        algorithm='auto',         # Let the algorithm choose the best method
        leaf_size=30,             # Leaf size for the tree structure
        p=2,                      # Use Euclidean distance
        metric='minkowski'        # Distance metric to use
    )
    knn_model.fit(X_train, y_train)

    y_pred_knn = knn_model.predict(X_test)

    # Evaluate the KNN model
    print("#############################################################################")
    print("K-Nearest Neighbors Classifier Model Evaluation:")
    print("Classification Report:\n", classification_report(y_test, y_pred_knn))
    print("Accuracy:", accuracy_score(y_test, y_pred_knn))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))
    print("5 values of original vs predicted:")
    print("Actual:", y_test.head().values, "\nPredicted:", y_pred_knn[:5])