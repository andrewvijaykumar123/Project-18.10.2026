from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    # Load the UCI Wine dataset
    X, y = load_wine(return_X_y=True, as_frame=True)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    # Train a Decision Tree classifier
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=[f"Class {i}" for i in range(3)])
    cm = confusion_matrix(y_test, y_pred)

    print("Decision Tree Wine Quality Prediction")
    print("=" * 38)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n", report)
    print("\nConfusion Matrix:\n", cm)

    # Save the decision tree plot for reference
    plt.figure(figsize=(18, 10))
    plot_tree(model, feature_names=X.columns, class_names=[str(i) for i in range(3)], filled=True)
    plt.savefig("wine_decision_tree.png", dpi=200, bbox_inches="tight")
    print("\nSaved decision tree plot to wine_decision_tree.png")


if __name__ == "__main__":
    main()
