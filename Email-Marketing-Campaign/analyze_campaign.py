import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Define file paths
email_file = 'email_table.csv'
opened_file = 'email_opened_table.csv'
clicked_file = 'link_clicked_table.csv'

# Load the datasets
print("Loading data...")
try:
    email_df = pd.read_csv(email_file)
    opened_df = pd.read_csv(opened_file)
    clicked_df = pd.read_csv(clicked_file)
    print("Data loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading file: {e}. Make sure the CSV files are in the same directory as the script.")
    exit()

# --- Data Preparation ---
print("Preparing data...")
# Added 'opened' column
# Created a set of opened email IDs for faster lookup
opened_email_ids = set(opened_df['email_id'])
email_df['opened'] = email_df['email_id'].apply(lambda x: 1 if x in opened_email_ids else 0)

# Added 'clicked' column
# Created a set of clicked email IDs for faster lookup
clicked_email_ids = set(clicked_df['email_id'])
email_df['clicked'] = email_df['email_id'].apply(lambda x: 1 if x in clicked_email_ids else 0)

print("Data preparation complete.")
print("\n--- Initial Analysis ---")

# --- Calculate Overall Rates ---
total_emails_sent = len(email_df)
total_opened = email_df['opened'].sum()
total_clicked = email_df['clicked'].sum()

open_rate = (total_opened / total_emails_sent) * 100 if total_emails_sent > 0 else 0
click_rate_overall = (total_clicked / total_emails_sent) * 100 if total_emails_sent > 0 else 0
click_rate_opened = (total_clicked / total_opened) * 100 if total_opened > 0 else 0

print(f"Total emails sent: {total_emails_sent}")
print(f"Total emails opened: {total_opened}")
print(f"Total links clicked: {total_clicked}")
print(f"\nOverall Open Rate: {open_rate:.2f}%")
print(f"Overall Click Rate (based on total sent): {click_rate_overall:.2f}%")
print(f"Click-Through Rate (based on opened emails): {click_rate_opened:.2f}%")


print("\n--- Combined Data Info ---")
print(email_df.info())
print("\n--- First 5 rows of combined data ---")
print(email_df.head())

# --- Model Building ---
print("\n--- Building Click Prediction Model ---")

# Excluded email_id (identifier) and opened (potential data leakage for predicting clicks *before* open)
features = ['email_text', 'email_version', 'hour', 'weekday', 'user_country', 'user_past_purchases']
target = 'clicked'

X = email_df[features]
y = email_df[target]

categorical_features = ['email_text', 'email_version', 'weekday', 'user_country']
numerical_features = ['hour', 'user_past_purchases'] # 'hour' could be treated cyclically, but let's start simple

# building a pipeline that tells sklearn how to transform each type of column.
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features) # drop='first' to avoid multicollinearity
    ],
    remainder='drop' # Drop any columns not specified
)

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")


model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000))])


print("Training the model...")
model_pipeline.fit(X_train, y_train)
print("Model training complete.")

# predictions on the test set
print("Evaluating the model...")
y_pred = model_pipeline.predict(X_test)
y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1] # Probabilities for the positive class (clicked=1)

# Evaluating the model
print("\n--- Model Evaluation ---")
print("Classification Report:")
print(classification_report(y_test, y_pred))

auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"AUC Score: {auc_score:.4f}")

# --- Feature Importance (for Logistic Regression) ---
feature_names_out = model_pipeline.named_steps['preprocessor'].get_feature_names_out()
coefficients = model_pipeline.named_steps['classifier'].coef_[0]

# Combine feature names and coefficients
feature_importance = pd.DataFrame({'feature': feature_names_out, 'coefficient': coefficients})
# absolute coefficients for ranking importance
feature_importance['abs_coefficient'] = np.abs(feature_importance['coefficient'])
feature_importance = feature_importance.sort_values(by='abs_coefficient', ascending=False)

print("\n--- Feature Importance (Logistic Regression Coefficients) ---")
print(feature_importance[['feature', 'coefficient']].head(10)) # Displaying top 10 features

# --- Segment Analysis (Example: Click rate by country) ---
print("\n--- Segment Analysis Examples ---")
print("\nClick Rate by Country:")
country_click_rate = email_df.groupby('user_country')['clicked'].mean().sort_values(ascending=False) * 100
print(country_click_rate)

print("\nClick Rate by Email Version:")
version_click_rate = email_df.groupby('email_version')['clicked'].mean().sort_values(ascending=False) * 100
print(version_click_rate)

print("\nClick Rate by Email Text Length:")
text_click_rate = email_df.groupby('email_text')['clicked'].mean().sort_values(ascending=False) * 100
print(text_click_rate)

print("\nClick Rate by Weekday:")
weekday_click_rate = email_df.groupby('weekday')['clicked'].mean().sort_values(ascending=False) * 100
print(weekday_click_rate)

# --- Visualizations ---
print("\n--- Generating Visualizations ---")

output_dir = 'plots'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print(f"Plots will be saved in the '{output_dir}/' directory.")

sns.set_style("whitegrid")
# 1. Bar plot for Click Rate by Country
plt.figure(figsize=(8, 5))
sns.barplot(x=country_click_rate.index, y=country_click_rate.values, palette="viridis")
plt.title('Click Rate by User Country')
plt.ylabel('Click Rate (%)')
plt.xlabel('User Country')
plt.tight_layout()
# plt.savefig(os.path.join(output_dir, 'click_rate_by_country.png'))
plt.close() # Close the plot to free memory

# 2. Bar plot for Click Rate by Email Version
plt.figure(figsize=(6, 4))
sns.barplot(x=version_click_rate.index, y=version_click_rate.values, palette="viridis")
plt.title('Click Rate by Email Version')
plt.ylabel('Click Rate (%)')
plt.xlabel('Email Version')
plt.tight_layout()
# plt.savefig(os.path.join(output_dir, 'click_rate_by_version.png'))
plt.close()

# 3. Bar plot for Click Rate by Email Text Length
plt.figure(figsize=(6, 4))
sns.barplot(x=text_click_rate.index, y=text_click_rate.values, palette="viridis")
plt.title('Click Rate by Email Text Length')
plt.ylabel('Click Rate (%)')
plt.xlabel('Email Text Length')
plt.tight_layout()
# plt.savefig(os.path.join(output_dir, 'click_rate_by_text.png'))
plt.close()

# 4. Bar plot for Click Rate by Weekday
plt.figure(figsize=(10, 5))
# Ensure weekdays are ordered correctly
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
sns.barplot(x=weekday_click_rate.index, y=weekday_click_rate.values, order=weekday_order, palette="viridis")
plt.title('Click Rate by Weekday')
plt.ylabel('Click Rate (%)')
plt.xlabel('Weekday')
plt.tight_layout()
# plt.savefig(os.path.join(output_dir, 'click_rate_by_weekday.png'))
plt.close()

# 5. Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(7, 7))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.tight_layout()
# plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
plt.close()

# 6. Distribution of Past Purchases for Clickers vs Non-Clickers
plt.figure(figsize=(10, 6))
sns.histplot(data=email_df, x='user_past_purchases', hue='clicked', kde=True, bins=30, palette="viridis")
plt.title('Distribution of Past Purchases by Click Status')
plt.xlabel('Number of Past Purchases')
plt.xlim(0, email_df['user_past_purchases'].quantile(0.99)) 
plt.tight_layout()
# plt.savefig(os.path.join(output_dir, 'past_purchases_distribution.png'))
plt.close()


print("Visualizations generated successfully.")
print("\nScript finished.")
