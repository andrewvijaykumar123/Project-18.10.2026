from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def evaluate_model(model_name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{model_name} Results")
    print("-" * 40)
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred, target_names=[f"Class {i}" for i in range(3)]))
    return accuracy


def main():
    X, y = load_wine(return_X_y=True, as_frame=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    decision_tree = DecisionTreeClassifier(random_state=42)
    random_forest = RandomForestClassifier(n_estimators=200, random_state=42)

    dt_acc = evaluate_model("Decision Tree", decision_tree, X_train, X_test, y_train, y_test)
    rf_acc = evaluate_model("Random Forest", random_forest, X_train, X_test, y_train, y_test)

    print("\nModel Comparison")
    print("-" * 40)
    print(f"Decision Tree Accuracy: {dt_acc:.4f}")
    print(f"Random Forest Accuracy: {rf_acc:.4f}")

    if rf_acc > dt_acc:
        print("Random Forest performed better on this dataset.")
    elif dt_acc > rf_acc:
        print("Decision Tree performed better on this dataset.")
    else:
        print("Both models performed equally well.")


if __name__ == "__main__":
    main()
