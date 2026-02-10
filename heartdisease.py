import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import randint, uniform

print("Loading dataset...")
df = pd.read_csv('heart_2020_cleaned.csv')
print(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns\n")

df.drop_duplicates(inplace=True)

def remove_outliers(df, cols):
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df

num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
df = remove_outliers(df, num_cols)
print(f"After preprocessing: {len(df)} rows\n")

print("="*70)
print("EXPLORATORY DATA ANALYSIS")
print("="*70 + "\n")

plt.figure(figsize=(8, 5))
sns.countplot(x='HeartDisease', data=df)
plt.title('Distribution of Heart Disease', fontweight='bold')
plt.savefig('eda_target_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

plt.figure(figsize=(14, 6))
sns.countplot(x='AgeCategory', hue='HeartDisease', data=df)
plt.title('Heart Disease by Age Category', fontweight='bold')
plt.xlabel('Age Category')
plt.xticks(rotation=45)
plt.legend(title='Heart Disease')
plt.tight_layout()
plt.savefig('eda_heartdisease_by_age.png', dpi=150, bbox_inches='tight')
plt.close()

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
fig = df[numeric_cols].hist(figsize=(16, 12), bins=20, edgecolor='black')
plt.suptitle("Distribution of Numeric Features", fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('eda_numeric_distributions.png', dpi=150, bbox_inches='tight')
plt.close()

num_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
n_features = len(num_features)
fig, axes = plt.subplots(2, (n_features + 1) // 2, figsize=(18, 10))
axes = axes.flatten()
for i, col in enumerate(num_features):
    sns.boxplot(y=df[col], ax=axes[i], color='skyblue')
    axes[i].set_title(f'Boxplot of {col}', fontweight='bold')
for j in range(i + 1, len(axes)):
    axes[j].axis('off')
plt.tight_layout()
plt.savefig('eda_boxplots_outliers.png', dpi=150, bbox_inches='tight')
plt.close()

df_encoded = df.copy()
le_temp = LabelEncoder()
for col in df_encoded.columns:
    if df_encoded[col].dtype == 'object':
        df_encoded[col] = le_temp.fit_transform(df_encoded[col])

plt.figure(figsize=(16, 14))
correlation_matrix = df_encoded.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Correlation Heatmap', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

target_corr = correlation_matrix['HeartDisease'].sort_values(ascending=False).drop('HeartDisease')
plt.figure(figsize=(10, 8))
target_corr.plot(kind='barh', color='coral')
plt.title('Feature Correlation with Heart Disease', fontsize=14, fontweight='bold')
plt.xlabel('Correlation Coefficient')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('eda_target_correlations.png', dpi=150, bbox_inches='tight')
plt.close()

key_features = ['Sex', 'GenHealth', 'Diabetic', 'PhysicalActivity', 'Smoking']
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
for i, feature in enumerate(key_features):
    if feature in df.columns:
        sns.countplot(x=feature, hue='HeartDisease', data=df, ax=axes[i])
        axes[i].set_title(f'Heart Disease by {feature}', fontweight='bold')
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].legend(title='Heart Disease')
axes[-1].axis('off')
plt.tight_layout()
plt.savefig('eda_categorical_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

print("EDA visualizations saved (7 PNG files)\n")

print("="*70)
print("MODEL TRAINING")
print("="*70 + "\n")

encoders = {}
label_mappings = {}
target_col = 'HeartDisease'
y = df[target_col]

le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)
encoders['HeartDisease'] = le_target
label_mappings['HeartDisease'] = dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))

X = df.drop(target_col, axis=1).copy()
for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le
        label_mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))

print(f"Features: {len(X.columns)}")
print(f"{list(X.columns)}\n")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Naive Bayes": GaussianNB(),
    "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss')
}

print("Base Model Performance:")
print("-" * 70)
best_accuracy = 0
best_model_name = None

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name}: {acc:.4f}")
    if acc > best_accuracy:
        best_accuracy = acc
        best_model_name = name

print(f"\nBest Base Model: {best_model_name} ({best_accuracy:.4f})\n")

print("Hyperparameter Tuning...")

rf_params = {
    "n_estimators": randint(200, 500),
    "max_depth": randint(5, 20),
    "min_samples_split": randint(2, 10),
    "min_samples_leaf": randint(1, 5),
    "max_features": ["sqrt", "log2"],
    "bootstrap": [True, False]
}

rf_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=rf_params,
    n_iter=20,
    scoring="accuracy",
    cv=3,
    random_state=42,
    n_jobs=-1,
    verbose=0
)
rf_search.fit(X_train, y_train)
best_rf = rf_search.best_estimator_
rf_acc = accuracy_score(y_test, best_rf.predict(X_test))

xgb_params = {
    "n_estimators": randint(200, 500),
    "max_depth": randint(3, 10),
    "learning_rate": uniform(0.01, 0.2),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.5, 0.5),
    "gamma": uniform(0, 3),
    "min_child_weight": randint(1, 7)
}

xgb_search = RandomizedSearchCV(
    XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42),
    param_distributions=xgb_params,
    n_iter=20,
    scoring="accuracy",
    cv=3,
    random_state=42,
    n_jobs=-1,
    verbose=0
)
xgb_search.fit(X_train, y_train)
best_xgb = xgb_search.best_estimator_
xgb_acc = accuracy_score(y_test, best_xgb.predict(X_test))

print(f"Random Forest (Tuned): {rf_acc:.4f}")
print(f"XGBoost (Tuned): {xgb_acc:.4f}\n")

if rf_acc >= xgb_acc:
    final_model = best_rf
    final_acc = rf_acc
    final_name = "Random Forest (Tuned)"
else:
    final_model = best_xgb
    final_acc = xgb_acc
    final_name = "XGBoost (Tuned)"

print("="*70)
print(f"Final Model: {final_name}")
print(f"Accuracy: {final_acc:.4f}")
print("="*70 + "\n")

y_pred_final = final_model.predict(X_test)
print(classification_report(y_test, y_pred_final, 
                          target_names=['No Heart Disease', 'Heart Disease']))

joblib.dump(final_model, "heartdisease_model.pkl")
joblib.dump(scaler, "scaler.pkl")
encoder_data = {
    'encoders': encoders,
    'label_mappings': label_mappings,
    'feature_names': list(X.columns)
}
joblib.dump(encoder_data, "encoders.pkl")

print("\nModel files saved: heartdisease_model.pkl, scaler.pkl, encoders.pkl")
print("EDA visualizations saved: 7 PNG files")
print("\nTraining complete!")