import pandas as pd
import numpy as np
import os
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

PHASE2_CSV = "../Results/phase2_output.csv"
TRAIN_CSV = "../Results/phase3_output_train.csv"
TEST_CSV = "../Results/phase3_output_test.csv"

# Load and preprocess the dataset, including balancing by class and scaling
def load_and_preprocess():
    if not os.path.exists(PHASE2_CSV):
        raise FileNotFoundError(f"Input file not found: {PHASE2_CSV}")

    # Load data from Phase 2 output
    df = pd.read_csv(PHASE2_CSV)
    grouped = df.groupby('class')  # Group by target class

    train_df = pd.DataFrame()
    test_df = pd.DataFrame()

    # Fixed sample sizes per class for balance
    n_samples_train = 48
    n_samples_test = 12

    # Sample training and testing sets for each class
    for class_name, class_group in grouped:
        total_class_rows = len(class_group)
        if total_class_rows >= (n_samples_train + n_samples_test):
            sampled_train = class_group.sample(n=n_samples_train, random_state=42)
            train_df = pd.concat([train_df, sampled_train])
            remaining_test = class_group.drop(sampled_train.index)
            test_df = pd.concat([test_df, remaining_test])
        elif total_class_rows >= n_samples_test:
            sampled_test = class_group.sample(n=n_samples_test, random_state=42)
            test_df = pd.concat([test_df, sampled_test])
            remaining_train = class_group.drop(sampled_test.index)
            test_df = pd.concat([train_df, remaining_train])
        else:
            train_df = pd.concat([train_df, class_group])

    # Save the balanced datasets
    train_df.to_csv(TRAIN_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)

    print("Balanced training class counts:\\n", train_df['class'].value_counts())
    print("Balanced test class counts:\\n", test_df['class'].value_counts())

    # Select cluster ratio features only
    X_train = train_df[[col for col in train_df.columns if col.startswith('cluster')]]
    y_train = train_df['class']
    X_test = test_df[[col for col in test_df.columns if col.startswith('cluster')]]
    y_test = test_df['class']

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, df

# Train and evaluate a model, and generate metrics + confusion matrix
def train_and_evaluate(model, model_name, X_train, X_test, y_train, y_test, df):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\\n=== {model_name} Evaluation ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\\n", classification_report(y_test, y_pred, zero_division=0))

    results_df = pd.DataFrame({
        'sid': df.loc[y_test.index, 'sid'].values,
        'actual_class': y_test.reset_index(drop=True),
        'predicted_class': y_pred
    })
    results_df.to_csv(f"../Results/{model_name}_predictions.csv", index=False)

    # Confusion matrix visualization
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=sorted(y_test.unique()),
                yticklabels=sorted(y_test.unique()))
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"../Results/{model_name}_confusion_matrix.png")
    plt.close()

# Plot learning curve of training and validation cost
def plot_learning_curve(model, model_name, X, y, scoring='neg_log_loss'):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, scoring=scoring, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10), shuffle=True, random_state=42
    )

    # Convert from negative log loss
    train_mean = -np.mean(train_scores, axis=1)
    val_mean = -np.mean(val_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mean, 'o-', label='Training Cost')
    plt.plot(train_sizes, val_mean, 'o-', label='Validation Cost')
    plt.title(f'Learning Curve (Cost): {model_name}')
    plt.xlabel('Training Set Size')
    plt.ylabel('Log Loss')
    plt.ylim(0, max(max(train_mean), max(val_mean)) + 0.1)
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"../Results/{model_name}_learning_curve_cost.png")
    plt.close()
    
def main():
    X_train, X_test, y_train, y_test, df = load_and_preprocess()

    # Logistic Regression - Tuned with GridSearchCV
    print("Tuning Logistic Regression with GridSearchCV and class_weight='balanced'...\\n")
    logreg_param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l2'],
        'solver': ['liblinear']
    }

    logreg_grid = GridSearchCV(
        LogisticRegression(max_iter=1000, class_weight='balanced'),
        logreg_param_grid,
        cv=5
    )
    logreg_grid.fit(X_train, y_train)
    best_logreg = logreg_grid.best_estimator_
    print(f"Best Logistic Regression Params: {logreg_grid.best_params_}\\n")
    train_and_evaluate(best_logreg, "Logistic_Regression_Tuned", X_train, X_test, y_train, y_test, df)
    plot_learning_curve(best_logreg, "Logistic_Regression_Tuned", X_train, y_train)

    # KNN Classifier with k=6
    print("Training KNN with k=6 (no GridSearchCV)...\\n")
    best_knn = KNeighborsClassifier(n_neighbors=6)
    best_knn.fit(X_train, y_train)
    train_and_evaluate(best_knn, "KNN_k6", X_train, X_test, y_train, y_test, df)
    plot_learning_curve(best_knn, "KNN_k6", X_train, y_train)

    # Random Forest Classifier - Tuned with GridSearchCV
    print("Tuning Random Forest with GridSearchCV...\\n")
    rf_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }
    rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=5)
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_
    print(f"Best Random Forest Params: {rf_grid.best_params_}\\n")
    train_and_evaluate(best_rf, "Random_Forest_Tuned", X_train, X_test, y_train, y_test, df)
    plot_learning_curve(best_rf, "Random_Forest_Tuned", X_train, y_train)

    # MLP Classifier - Tuned with GridSearchCV
    print("Tuning MLP Classifier with GridSearchCV...\\n")
    mlp_param_grid = {
        'hidden_layer_sizes': [(40, 15)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001],
        'learning_rate_init': [0.001, 0.01]
    }

    mlp_grid = GridSearchCV(
        MLPClassifier(max_iter=500, random_state=42, early_stopping=True),
        mlp_param_grid,
        cv=5
    )
    mlp_grid.fit(X_train, y_train)
    best_mlp = mlp_grid.best_estimator_
    print(f"Best MLP Params: {mlp_grid.best_params_}\\n")
    train_and_evaluate(best_mlp, "MLP_Tuned", X_train, X_test, y_train, y_test, df)
    plot_learning_curve(best_mlp, "MLP_Tuned", X_train, y_train)

if __name__ == "__main__":
    main()
