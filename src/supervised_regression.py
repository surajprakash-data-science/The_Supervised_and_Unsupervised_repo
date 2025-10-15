from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score

if __name__ == "__main__":
    from data_ingestion import load_data, preprocess_data

    df = load_data("A:\ML practice\supervised and unsupervised learning\data\possum.csv")
    df = preprocess_data(df)

    variable_coloumn = df[[ "skullw", "totlngth"]]
    target_column = df["hdlngth"]

    X_train, X_test, y_train, y_test = train_test_split(variable_coloumn, target_column, test_size=0.2, random_state=42)

    # Train a simple linear regression model
    #======================================================================
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Evaluate the model
    print("#############################################################################")
    print("Linear Regression Model Evaluation:")
    print("Coefficients:", model.coef_)
    print("Intercept:", model.intercept_)
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    print("R² Score:", r2_score(y_test, y_pred))
    print("5 values of original vs predicted:")
    print("Actual:", y_test.head().values, "\nPredicted:", y_pred[:5])


    
    # Train a simple lasso(L1) regression model
    #======================================================================
    lasso_model = Lasso(alpha=0.1) # alpha controls regularization strength, higher values mean more regularization
    lasso_model.fit(X_train, y_train)

    y_pred_lasso = lasso_model.predict(X_test)

    # Evaluate the lasso model
    print("#############################################################################")
    print("Lasso(L1) Regression Model Evaluation:")
    print("Coefficients:", lasso_model.coef_)
    print("Intercept:", lasso_model.intercept_)
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred_lasso))
    print("R² Score:", r2_score(y_test, y_pred_lasso))
    print("5 values of original vs predicted:")
    print("Actual:", y_test.head().values, "\nPredicted:", y_pred_lasso[:5])

    # L1 regularization can shrink some coefficients to zero, effectively performing feature selection.
    # L2 regularization tends to shrink coefficients evenly but does not set any to zero.
    
    # Train a simple Ridge(L2) regression model
    #======================================================================
    ridge_model = Ridge(alpha=1.0) # alpha controls regularization strength, higher values mean more regularization
    ridge_model.fit(X_train, y_train)

    y_pred_ridge = ridge_model.predict(X_test)

    # Evaluate the ridge model
    print("#############################################################################")
    print("Ridge(L2) Regression Model Evaluation:")
    print("Coefficients:", ridge_model.coef_)
    print("Intercept:", ridge_model.intercept_)
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred_ridge))
    print("R² Score:", r2_score(y_test, y_pred_ridge))
    print("5 values of original vs predicted:")
    print("Actual:", y_test.head().values, "\nPredicted:", y_pred_ridge[:5])

    # Train a Desision Tree regression model
    #======================================================================
    tree_model = DecisionTreeRegressor(
        criterion="friedman_mse",   # other options: squared_error, absolute_error, poisson or gini for classification
        max_depth=3,                # maximum depth of the tree, used for overfitting control
        max_features=None,          # number of features to consider when looking for the best split
        min_samples_split=2,        # minimum number of samples required to split an internal node
        min_samples_leaf=1,         # minimum number of samples required to be at a leaf node
        random_state=42     
    )       
    tree_model.fit(X_train, y_train)

    y_pred_tree = tree_model.predict(X_test)

    # Evaluate the decision tree model
    print("#############################################################################")
    print("Decision Tree Regression Model Evaluation:")
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred_tree))
    print("R² Score:", r2_score(y_test, y_pred_tree))
    print("5 values of original vs predicted:")
    print("Actual:", y_test.head().values, "\nPredicted:", y_pred_tree[:5])

    # friedman_mse: use when trainig data has correlated features, it can lead to better performance.
    # squared_error: standard mean squared error, sensitive to outliers, default and most common.
    # absolute_error: use when your data has outliers, as it is more robust to them.
    # poisson: use for count data (e.g., number of events, visits, etc.), assumes target variable follows a Poisson distribution.


    # Train a Random Forest regressor model
    #======================================================================
    forest_model = RandomForestRegressor(
        n_estimators=10,           # number of trees in the forest
        criterion="squared_error", # other options: absolute_error, poisson
        max_depth=4,               # maximum depth of the tree
        max_features= None,        # number of features to consider when looking for the best split
        min_samples_split=2,       # minimum number of samples required to split an internal node
        min_samples_leaf=1,        # minimum number of samples required to be at a leaf node
        random_state=42,
        n_jobs=-1                  # use all available cores for parallel processing
    )
    forest_model.fit(X_train, y_train)

    y_pred_forest = forest_model.predict(X_test)

    # Evaluate the random forest model
    print("#############################################################################")
    print("Random Forest Regression Model Evaluation:")
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred_forest))
    print("R² Score:", r2_score(y_test, y_pred_forest))
    print("5 values of original vs predicted:")
    print("Actual:", y_test.head().values, "\nPredicted:", y_pred_forest[:5])



    # Train a Gradient Boosting regressor model
    #======================================================================
    gb_model = GradientBoostingRegressor(
        n_estimators=100,       # number of boosting stages to be run
        learning_rate=0.05,     # step size shrinkage used in update to prevent overfitting
        max_depth=3,            # maximum depth of the individual regression estimators
        loss="squared_error",   # other options: absolute_error, huber, quantile
        random_state=42
    )
    gb_model.fit(X_train, y_train)

    y_pred_gb = gb_model.predict(X_test)

    # Evaluate the gradient boosting model
    print("#############################################################################")  
    print("Gradient Boosting Regression Model Evaluation:")
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred_gb))
    print("R² Score:", r2_score(y_test, y_pred_gb))
    print("5 values of original vs predicted:")
    print("Actual:", y_test.head().values, "\nPredicted:", y_pred_gb[:5])


    # Train a XGBoost regressor model
    #======================================================================
    xg_model = XGBRegressor(
        n_estimators=100,       # 
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    xg_model.fit(X_train, y_train)

    y_pred_xg = xg_model.predict(X_test)

    # Evaluate the XGBoost model
    print("#############################################################################")  
    print("XGBoost Regression Model Evaluation:")
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred_xg))
    print("R² Score:", r2_score(y_test, y_pred_xg))
    print("5 values of original vs predicted:")
    print("Actual:", y_test.head().values, "\nPredicted:", y_pred_xg[:5])


    # Train a LightGBM regressor model
    #======================================================================
    lgbm_model = LGBMRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    lgbm_model.fit(X_train, y_train)
    
    y_pred_lgbm = lgbm_model.predict(X_test)

    # Evaluate the LightGBM model
    print("#############################################################################")
    print("LightGBM Regression Model Evaluation:")
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred_lgbm))
    print("R² Score:", r2_score(y_test, y_pred_lgbm))
    print("5 values of original vs predicted:")
    print("Actual:", y_test.head().values, "\nPredicted:", y_pred_lgbm[:5])