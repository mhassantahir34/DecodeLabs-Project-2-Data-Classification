import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ==========================================================
# 1. LOAD DATASET
# ==========================================================

data = pd.read_excel("dataset.xlsx")

print("=" * 70)
print("DECODELABS - PROJECT 2")
print("DATA CLASSIFICATION USING AI")
print("=" * 70)

print("\nDataset loaded successfully!")

print("\nFirst 5 rows:")
print(data.head())

print("\nDataset Shape:")
print(data.shape)

print("\nColumns:")
print(data.columns.tolist())

print("\nMissing Values:")
print(data.isnull().sum())

# ==========================================================
# 2. UNDERSTAND TARGET
# ==========================================================

target = "OrderStatus"

print("\nOrder Status Distribution:")
print(data[target].value_counts())

# ==========================================================
# 3. DATA PREPROCESSING
# ==========================================================

# Convert Date into useful numerical information
data["Date"] = pd.to_datetime(data["Date"])

data["Year"] = data["Date"].dt.year
data["Month"] = data["Date"].dt.month
data["Day"] = data["Date"].dt.day

data.drop("Date", axis=1, inplace=True)

# Remove columns that are only identifiers
columns_to_remove = [
    "OrderID",
    "CustomerID",
    "ShippingAddress",
    "TrackingNumber"
]
data.drop(
    columns=columns_to_remove,
    inplace=True
)

# Fill missing values
for column in data.columns:
    if pd.api.types.is_numeric_dtype(data[column]):
        data[column] = data[column].fillna(
            data[column].median()
        )
    else:
        data[column] = data[column].fillna(
            data[column].mode()[0]
        )

# ==========================================================
# 4. ENCODE CATEGORICAL DATA
# ==========================================================

encoders = {}

for column in data.select_dtypes(
    include=["object", "string"]
).columns:
    encoder = LabelEncoder()
    data[column] = encoder.fit_transform(
        data[column].astype(str)
    )
    encoders[column] = encoder

# ==========================================================
# 5. SEPARATE FEATURES AND TARGET
# ==========================================================

X = data.drop(
    target,
    axis=1
)
y = data[target]

print("\nFeatures used for classification:")
print(X.columns.tolist())

# ==========================================================
# 6. TRAIN / TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# ==========================================================
# 7. SCALE DATA FOR KNN AND LOGISTIC REGRESSION
# ==========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)
X_test_scaled = scaler.transform(
    X_test
)

# ==========================================================
# 8. CREATE CLASSIFICATION MODELS
# ==========================================================

models = {
    "Logistic Regression":
        LogisticRegression(
            max_iter=2000,
            random_state=42
        ),
    "Decision Tree":
        DecisionTreeClassifier(
            max_depth=8,
            random_state=42
        ),
    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ),
    "K-Nearest Neighbors":
        KNeighborsClassifier(
            n_neighbors=5
        )
}

# ==========================================================
# 9. TRAIN AND TEST MODELS
# ==========================================================

results = {}
trained_models = {}
predictions = {}

for name, model in models.items():

    print("\n" + "=" * 70)
    print("MODEL:", name)
    print("=" * 70)

    # KNN and Logistic Regression work better with scaled data
    if name in [
        "Logistic Regression",
        "K-Nearest Neighbors"
    ]:
        model.fit(
            X_train_scaled,
            y_train
        )
        prediction = model.predict(
            X_test_scaled
        )
    else:
        model.fit(
            X_train,
            y_train
        )
        prediction = model.predict(
            X_test
        )

    # Calculate performance
    accuracy = accuracy_score(
        y_test,
        prediction
    )
    precision = precision_score(
        y_test,
        prediction,
        average="weighted",
        zero_division=0
    )
    recall = recall_score(
        y_test,
        prediction,
        average="weighted",
        zero_division=0
    )
    f1 = f1_score(
        y_test,
        prediction,
        average="weighted",
        zero_division=0
    )

    results[name] = {

        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }

    trained_models[name] = model
    predictions[name] = prediction

    print(
        f"Accuracy : {accuracy * 100:.2f}%"
    )
    print(
        f"Precision: {precision * 100:.2f}%"
    )
    print(
        f"Recall   : {recall * 100:.2f}%"
    )
    print(
        f"F1 Score : {f1 * 100:.2f}%"
    )
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            prediction,
            zero_division=0
        )
    )

# ==========================================================
# 10. MODEL COMPARISON
# ==========================================================

results_df = pd.DataFrame(results).T

results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
)

print("\n" + "=" * 70)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 70)
print(
    results_df.to_string()
)

# Save results
results_df.to_csv(
    "model_comparison.csv"
)

# ==========================================================
# 11. FIND BEST MODEL
# ==========================================================

best_model_name = results_df.index[0]
best_model = trained_models[
    best_model_name
]

best_prediction = predictions[
    best_model_name
]

best_accuracy = results_df.loc[
    best_model_name,
    "Accuracy"
]

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)
print(
    "Best Model:",
    best_model_name
)
print(
    f"Best Accuracy: {best_accuracy * 100:.2f}%"
)

# ==========================================================
# 12. CONFUSION MATRIX
# ==========================================================

matrix = confusion_matrix(
    y_test,
    best_prediction
)
display = ConfusionMatrixDisplay(
    confusion_matrix=matrix
)
display.plot()
plt.title(
    "Confusion Matrix - " + best_model_name
)
plt.tight_layout()
plt.savefig(
    "confusion_matrix.png",
    dpi=150
)
plt.show()

# ==========================================================
# 13. FEATURE IMPORTANCE
# ==========================================================

if best_model_name in [
    "Decision Tree",
    "Random Forest"
]:

    importance = best_model.feature_importances_
    feature_importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importance
    })

    feature_importance = feature_importance.sort_values(
        by="Importance",
        ascending=False
    )

    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE")
    print("=" * 70)
    print(
        feature_importance.to_string(
            index=False
        )
    )

    # Plot top features
    top_features = feature_importance.head(10)
    plt.figure(figsize=(10, 6))
    plt.barh(
        top_features["Feature"],
        top_features["Importance"]
    )
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(
        "Top 10 Important Features"
    )
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(
        "feature_importance.png",
        dpi=150
    )
    plt.show()

# ==========================================================
# 14. TEST WITH NEW / UNSEEN DATA
# ==========================================================

print("\n" + "=" * 70)
print("NEW DATA PREDICTION")
print("=" * 70)

new_order = pd.DataFrame([{
    "Product": "Laptop",
    "Quantity": 2,
    "UnitPrice": 800.00,
    "PaymentMethod": "Credit Card",
    "ItemsInCart": 4,
    "CouponCode": "SAVE10",
    "ReferralSource": "Instagram",
    "TotalPrice": 1600.00,
    "Year": 2026,
    "Month": 8,
    "Day": 8
}])

# Encode new categorical values
for column in new_order.columns:
    if column in encoders:
        encoder = encoders[column]
        value = new_order[column].iloc[0]
        if value in encoder.classes_:
            new_order[column] = encoder.transform(
                new_order[column].astype(str)
            )
        else:
            new_order[column] = 0

# Make sure feature order is same
new_order = new_order[
    X.columns
]

# Predict using correct preprocessing
if best_model_name in [
    "Logistic Regression",
    "K-Nearest Neighbors"
]:
    new_order_scaled = scaler.transform(
        new_order
    )
    new_prediction = best_model.predict(
        new_order_scaled
    )
else:
    new_prediction = best_model.predict(
        new_order
    )

# Convert prediction back to original status
target_encoder = encoders[target]

predicted_status = target_encoder.inverse_transform(
    new_prediction
)[0]

print("\nNew Order:")
print(
    new_order.to_string(
        index=False
    )
)
print(
    "\nPredicted Order Status:",
    predicted_status
)

# ==========================================================
# 15. FINAL MESSAGE
# ==========================================================

print("\n" + "=" * 70)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 70)
print("\nGenerated files:")
print("1. model_comparison.csv")
print("2. confusion_matrix.png")
print("3. feature_importance.png")
print(
    "\nThe classification project is complete."
)