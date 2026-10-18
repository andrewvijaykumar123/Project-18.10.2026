from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def print_dataset_summary(X, y):
    print("Wine dataset summary")
    print("-" * 40)
    print(f"Samples: {X.shape[0]}")
    print(f"Features: {X.shape[1]}")
    print(f"Classes: {sorted(y.unique().tolist())}")
    print("Class distribution:")
    print(pd.Series(y).value_counts().sort_index())
    print()


def evaluate_classifier(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        target_names=[f"Class {i}" for i in range(3)],
        output_dict=True,
    )
    cm = confusion_matrix(y_test, y_pred)

    print(f"{name} evaluation")
    print("-" * 40)
    print(f"Test accuracy: {accuracy:.4f}")
    print("Per-class F1 scores:")
    for label in range(3):
        print(f"  Class {label}: {report[f'Class {label}']['f1-score']:.4f}")
    print("\nConfusion matrix:\n", cm)
    print()

    return accuracy, model, cm


def print_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        importance = pd.Series(model.feature_importances_, index=feature_names)
        top_features = importance.sort_values(ascending=False).head(5)
        print("Top 5 most important features:")
        print(top_features)
        print()


def save_tree_plot(model, feature_names):
    fig, ax = plt.subplots(figsize=(18, 10))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=[str(i) for i in range(3)],
        filled=True,
        ax=ax,
    )
    fig.savefig("wine_decision_tree_enhanced.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved enhanced decision tree plot to wine_decision_tree_enhanced.png")


def main():
    X, y = load_wine(return_X_y=True, as_frame=True)
    feature_names = list(X.columns)

    print_dataset_summary(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    decision_tree = DecisionTreeClassifier(random_state=42, max_depth=None)
    random_forest = RandomForestClassifier(n_estimators=300, random_state=42)

    dt_accuracy, dt_model, _ = evaluate_classifier(
        "Decision Tree",
        decision_tree,
        X_train,
        X_test,
        y_train,
        y_test,
    )
    rf_accuracy, _, _ = evaluate_classifier(
        "Random Forest",
        random_forest,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    print_feature_importance(dt_model, feature_names)
    save_tree_plot(dt_model, feature_names)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(dt_model, X, y, cv=cv, scoring="accuracy")
    print("5-fold cross-validation accuracy for Decision Tree:")
    print(f"Mean: {cv_scores.mean():.4f}")
    print(f"Std: {cv_scores.std():.4f}")
    print(f"Scores: {cv_scores}")

    print("\nFinal comparison")
    print("-" * 40)
    print(f"Decision Tree test accuracy: {dt_accuracy:.4f}")
    print(f"Random Forest test accuracy: {rf_accuracy:.4f}")
    print("Decision Tree accomplished strong performance and remains interpretable.")
    print("Random Forest showed slightly stronger predictive power on this split.")


if __name__ == "__main__":
    main()
