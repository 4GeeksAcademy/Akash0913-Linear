# Importing Libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression

# Load Data
data_url = "https://raw.githubusercontent.com/4GeeksAcademy/linear-regression-project-tutorial/main/medical_insurance_cost.csv"
total_data = pd.read_csv(data_url)
total_data = total_data.drop_duplicates().reset_index(drop=True)
print(total_data.head())

# Exploratory Data Analysis (EDA)
plt.figure(figsize=(10, 6))
sns.heatmap(total_data.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Pair plot for numerical variables
sns.pairplot(total_data, hue="smoker")
plt.show()

# Encode Categorical Variables
total_data["sex_n"] = pd.factorize(total_data["sex"])[0]
total_data["smoker_n"] = pd.factorize(total_data["smoker"])[0]
total_data["region_n"] = pd.factorize(total_data["region"])[0]

# Split Data Before Scaling
X = total_data[["age", "bmi", "children", "sex_n", "smoker_n", "region_n"]]
y = total_data["charges"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale Features
scaler = MinMaxScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

# Feature Selection
selector = SelectKBest(score_func=f_regression, k=4)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

selected_columns = X_train.columns[selector.get_support()]
print(f"Selected Features: {selected_columns}")

# Build and Train the Linear Regression Model
model = LinearRegression()
model.fit(X_train_selected, y_train)

print(f"Intercept: {model.intercept_}")
print(f"Coefficients: {model.coef_}")

# Evaluate Model Performance
y_pred = model.predict(X_test_selected)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")
