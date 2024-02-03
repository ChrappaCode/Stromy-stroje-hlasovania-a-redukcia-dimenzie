import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.svm import SVR
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import plotly.express as px
from sklearn.decomposition import PCA

#ZDROJE: Moje zadanie 1 a ChatGPT :D

pd.options.display.width = None
pd.options.display.max_columns = None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)

pd.set_option('mode.chained_assignment', None)

df = pd.read_csv('zadanie2_dataset.csv')
df_eda = pd.read_csv('zadanie2_dataset.csv')

#19200 veci tam je

#------------------------------------Outliery--------------------------------------
df.dropna(inplace=True)
df['Mileage'] = df['Mileage'].str.extract(r'(\d+)').astype(int)

df_eda.dropna(inplace=True)
df_eda = df_eda.drop_duplicates()


df.drop(columns=['ID'], inplace=True)
df.drop(columns=['Doors'], inplace=True)
df.drop(columns=['Levy'], inplace=True)
df.drop(columns=['Model'], inplace=True)
df.drop(columns=['Color'], inplace=True)

df = df[df['Manufacturer'] != 'სხვა']

df = df.drop_duplicates()

df = df[df['Mileage'] < 400000]
df = df[df['Prod. year'] > 1950]
df = df[df['Engine volume'] < 19]
df_eda = df_eda[df_eda['Engine volume'] < 19]
df = df[(df['Price'] >= 1000) & (df['Price'] <= 250000)]
#------------------------------------------------------------------------------------


#------------------------------------Kodovanie---------------------------------------
le = LabelEncoder()
df['Leather interior'] = le.fit_transform(df['Leather interior'])
df['Turbo engine'] = le.fit_transform(df['Turbo engine'])
df['Left wheel'] = le.fit_transform(df['Left wheel'])

dummies = pd.get_dummies(df['Fuel type'])
dummies = dummies.astype(int)
df = pd.concat([df.drop('Fuel type', axis=1), dummies], axis=1)

dummies = pd.get_dummies(df['Drive wheels'])
dummies = dummies.astype(int)
df = pd.concat([df.drop('Drive wheels', axis=1), dummies], axis=1)

dummies = pd.get_dummies(df['Manufacturer'])
dummies = dummies.astype(int)
df = pd.concat([df.drop('Manufacturer', axis=1), dummies], axis=1)

dummies = pd.get_dummies(df['Gear box type'])
dummies = dummies.astype(int)
df = pd.concat([df.drop('Gear box type', axis=1), dummies], axis=1)

dummies = pd.get_dummies(df['Category'])
dummies = dummies.astype(int)
df = pd.concat([df.drop('Category', axis=1), dummies], axis=1)
#-------------------------------------------------------------------------------------



#--------------------------------Rozdelenie na vstup a vystup / normalizacia--------------------------------------------

X = df.drop(columns=['Price'])  # vstup
y = df['Price']                 # výstup

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42) # trenovacia a testovacia 0,02 na forest, 0,15 na tree, test_size=0.0007 svm

# Print dataset shapes
print("*"*100, "Dataset shapes", "*"*100)
print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_test: {y_test.shape}")

scaler = StandardScaler()               # normalizácia
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)

X_train.hist(bins=50, figsize=(20, 15))
plt.suptitle('Histograms after scaling/standardizing')
plt.show()
#-------------------------------------------------------------------------------------------------------------------------


#-------------------------------------------------DecisionTreeRegressor - With residuals-----------------------------------------------------------
tree_model = DecisionTreeRegressor(max_depth=9, random_state=42)

tree_model.fit(X_train, y_train)

plt.figure(figsize=(19.2, 10.8))
plot_tree(tree_model, filled=True, feature_names=X_train.columns)
plt.show()

y_train_pred = tree_model.predict(X_train)

y_test_pred = tree_model.predict(X_test)

r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

print("----------------------Tree----------------------------")
print(f"R2 Score (Train): {r2_train}")
print(f"R2 Score (Test): {r2_test}")
print(f"Root Mean Squared Error (Train): {rmse_train}")
print(f"Root Mean Squared Error (Test): {rmse_test}")

train_residuals = y_train - y_train_pred
test_residuals = y_test - y_test_pred

plt.figure(figsize=(12, 6))

# Residual plot for the training set
plt.subplot(1, 2, 1)
plt.scatter(y_train_pred, train_residuals, c='blue', label='Training Data')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.axhline(y=0, color='k', linestyle='--')
plt.title("Residual Plot (Training Set)")
plt.legend()

# Residual plot for the test set
plt.subplot(1, 2, 2)
plt.scatter(y_test_pred, test_residuals, c='red', label='Test Data')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.axhline(y=0, color='k', linestyle='--')
plt.title("Residual Plot (Test Set)")
plt.legend()

plt.tight_layout()
plt.show()
#-------------------------------------------------------------------------------------------------------------------------


#-----------------------------------------RandomForestRegressor - With residuals----------------------------------------------------------------
rf_model = RandomForestRegressor(max_depth=9, random_state=42)

rf_model.fit(X_train, y_train)

y_train_pred_rf = rf_model.predict(X_train)

y_test_pred_rf = rf_model.predict(X_test)

# Calculate R2 scores for training and test sets
r2_train_rf = r2_score(y_train, y_train_pred_rf)
r2_test_rf = r2_score(y_test, y_test_pred_rf)

# Calculate Mean Squared Error for training and test sets

rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred_rf))
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred_rf))

importances = rf_model.feature_importances_
feature_names = X.columns
indices = importances.argsort()[::-1]

# Select the top N most important features to visualize
top_n = 10  # Set N to the number of top features you want to display

# Plot the top N most important features with red bars
plt.figure(figsize=(19, 10))
plt.title(f"Top {top_n} Feature Importances")
plt.bar(range(top_n), importances[indices][:top_n], align="center", color='red')
plt.xticks(range(top_n), [feature_names[i] for i in indices][:top_n], rotation=90)
plt.show()

print("----------------------Forest------------------------")
print(f"R2 Score (Train - Random Forest): {r2_train_rf}")
print(f"R2 Score (Test - Random Forest): {r2_test_rf}")
print(f"Root Mean Squared Error (Train - Random Forest): {rmse_train}")
print(f"Root Mean Squared Error (Test - Random Forest): {rmse_test}")

# Create residual plots
train_residuals_rf = y_train - y_train_pred_rf
test_residuals_rf = y_test - y_test_pred_rf

plt.figure(figsize=(15, 10))
plot_tree(rf_model.estimators_[0], filled=True, feature_names=X_train.columns)
plt.show()

plt.figure(figsize=(12, 6))

# Residual plot for the training set
plt.subplot(1, 2, 1)
plt.scatter(y_train_pred_rf, train_residuals_rf, c='blue', label='Training Data')
plt.xlabel("Predicted Values (Random Forest)")
plt.ylabel("Residuals (Random Forest)")
plt.axhline(y=0, color='k', linestyle='--')
plt.title("Residual Plot (Training Set - Random Forest)")
plt.legend()

# Residual plot for the test set
plt.subplot(1, 2, 2)
plt.scatter(y_test_pred_rf, test_residuals_rf, c='red', label='Test Data')
plt.xlabel("Predicted Values (Random Forest)")
plt.ylabel("Residuals (Random Forest)")
plt.axhline(y=0, color='k', linestyle='--')
plt.title("Residual Plot (Test Set - Random Forest)")
plt.legend()

plt.tight_layout()
plt.show()
#-------------------------------------------------------------------------------------------------------------------------


#---------------------------------------------SVM residuals-----------------------------------------------------------
svm_model = SVR(kernel='rbf', C=1300, gamma=1)  # You can choose the appropriate kernel for your problem

# Fit the model on the training data
svm_model.fit(X_train, y_train)

# Make predictions on the training set
y_train_pred_svm = svm_model.predict(X_train)

# Make predictions on the test set
y_test_pred_svm = svm_model.predict(X_test)
r2_train_svm = r2_score(y_train, y_train_pred_svm)
r2_test_svm = r2_score(y_test, y_test_pred_svm)


rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred_svm))
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred_svm))


print("----------------------SVM----------------------")
print(f"R2 Score (Train - SVM): {r2_train_svm}")
print(f"R2 Score (Test - SVM): {r2_test_svm}")
print(f"Root Mean Squared Error (Train - SVM): {rmse_train}")
print(f"Root Mean Squared Error (Test - SVM): {rmse_test}")

train_residuals_svm = y_train - y_train_pred_svm
test_residuals_svm = y_test - y_test_pred_svm

plt.figure(figsize=(12, 6))

# Residual plot for the training set
plt.subplot(1, 2, 1)
plt.scatter(y_train_pred_svm, train_residuals_svm, c='blue', label='Training Data')
plt.xlabel("Predicted Values (SVM)")
plt.ylabel("Residuals (SVM)")
plt.axhline(y=0, color='k', linestyle='--')
plt.title("Residual Plot (Training Set - SVM)")
plt.legend()

# Residual plot for the test set
plt.subplot(1, 2, 2)
plt.scatter(y_test_pred_svm, test_residuals_svm, c='red', label='Test Data')#
plt.xlabel("Predicted Values (SVM)")
plt.ylabel("Residuals (SVM)")
plt.axhline(y=0, color='k', linestyle='--')
plt.title("Residual Plot (Test Set - SVM)")
plt.legend()

plt.tight_layout()
plt.show()
#-------------------------------------------------------------------------------------------------------------------------

#EDA----------------------------------------------------------------------------------------------------------------------
labels = df_eda['Color'].unique()  # Get unique values from the column
sizes = df_eda['Color'].value_counts()  # Count occurrences of each unique value

plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, shadow=True)
plt.title('Pie Chart of car Color')
plt.axis('equal')
plt.show()


plt.figure(figsize=(10, 6))
sns.violinplot(x="Doors", y="Category", data=df_eda, inner="stick", color="yellow")
plt.title("Violin Plot of Doors by Category")
plt.show()

df_eda = df_eda[df_eda['Prod. year'] > 1979]

plt.figure(figsize=(10, 6))
sns.violinplot(x="Cylinders", y="Prod. year", data=df_eda, inner="stick", color="blue")
plt.title("Violin Plot of Cylinders by Prod. year")
plt.show()

df_eda = df_eda[df_eda['Price'] < 150000]
fig = px.scatter(df_eda, x='Engine volume', y='Prod. year', color='Price',
                 size='Engine volume', hover_data=['Leather interior', 'Model'])
fig.update_layout(title='Interactive Scatter Plot showing Category to Prod. year and coloring to Price on hover showing Leather interior and Model')
fig.show()
#-------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------Scatter--------------------------------------------------------------------------

X1 = df["Cylinders"]
X2 = df["Airbags"]
X3 = df["Mileage"]

# y je vektor s cenami
y = df['Price']

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Nastavenie farieb podľa cien
colors = cm.viridis(y / max(y))  # Prispôsobte farbu podľa rozsahu cien

ax.scatter(X1, X2, X3, c=colors, marker='o')

ax.set_xlabel('Cylinders')
ax.set_ylabel('Airbags ')
ax.set_zlabel('Mileage')

plt.show()

#--

X1 = df["Cylinders"]
X2 = df["Airbags"]
X3 = df["Mileage"]
y = df['Price']

fig = px.scatter_3d(df, x=X1, y=X2, z=X3, color=y, opacity=0.7)
fig.update_traces(marker=dict(size=5))
fig.update_layout(scene=dict(xaxis_title='Cylinders', yaxis_title='Airbags', zaxis_title='Mileage'))
fig.show()
#-------------------------------------------------------------------------------------------------------------------------


#----------------------------------------------PCA------------------------------------------------------------------------
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

df_pca = pd.DataFrame(data=X_pca, columns=['Cylinders', 'Airbags', 'Mileage'])

df_pca['Price'] = y

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

colors = df_pca['Price']
scatter = ax.scatter(df_pca['Cylinders'], df_pca['Airbags'], df_pca['Mileage'], c=colors, cmap='viridis')

ax.set_xlabel('Cylinders')
ax.set_ylabel('Airbags')
ax.set_zlabel('Mileage')
plt.title("3D Scatter Plot with Price Color")

cbar = plt.colorbar(scatter)
cbar.set_label('Price')

plt.show()

#--

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

df_pca = pd.DataFrame(data=X_pca, columns=['Cylinders', 'Airbags', 'Mileage'])

df_pca['Price'] = y

fig = px.scatter_3d(df_pca, x='Cylinders', y='Airbags', z='Mileage', color='Price', color_continuous_scale='Viridis')
fig.update_layout(scene=dict(xaxis_title='Cylinders', yaxis_title='Airbags', zaxis_title='Mileage'))
fig.update_layout(title="Interactive 3D Scatter Plot with Price Color")
fig.show()
#-------------------------------------------------------------------------------------------------------------------------

#--------------------------------------COR MATRIX-----------------------------------------------------------------------------
corr_matrix = df.corr()

top_features = corr_matrix['Price'].abs().sort_values(ascending=False).index[:30]  # Choose the top n features

X_subset = df[top_features]
y = df['Price']

X_trainCOR, X_testCOR, y_trainCOR, y_testCOR = train_test_split(X_subset, y, test_size=0.2, random_state=42)
print("*"*100, "Dataset shapes COR ", "*"*100)
print(f"X_train: {X_trainCOR.shape}")
print(f"X_test: {X_testCOR.shape}")
print(f"y_train: {y_trainCOR.shape}")
print(f"y_test: {y_testCOR.shape}")

random_forest_model = RandomForestRegressor(n_estimators=100, max_depth=2, random_state=42)

random_forest_model.fit(X_trainCOR, y_trainCOR)

y_train_pred = random_forest_model.predict(X_trainCOR)
y_test_pred = random_forest_model.predict(X_testCOR)

r2_train = r2_score(y_trainCOR, y_train_pred)
r2_test = r2_score(y_testCOR, y_test_pred)

rmse_train = np.sqrt(mean_squared_error(y_trainCOR, y_train_pred))
rmse_test = np.sqrt(mean_squared_error(y_testCOR, y_test_pred))

print("----------------------Cor Matrix------------------------")
print(f"Cor Mat R2 Score (Train): {r2_train}")
print(f"Cor Mat R2 Score (Test): {r2_test}")
print(f"Cor Mat RMSE (Train): {rmse_train}")
print(f"Cor Mat RMSE (Test): {rmse_test}")
print("*"*100)


residuals_train = y_trainCOR - y_train_pred
residuals_test = y_testCOR - y_test_pred

plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.title("Residual Plot (Train - Cor Matrix)")
plt.scatter(y_train_pred, residuals_train, color='blue')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")

plt.subplot(1, 2, 2)
plt.title("Residual Plot (Test - Cor Matrix)")
plt.scatter(y_test_pred, residuals_test, color='green')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")

plt.tight_layout()
plt.show()
#-------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------PCA------------------------------------------------------------------
variances = X_train.var(axis=0)

variance_threshold = 0.2

selected_features = X_train.columns[variances >= variance_threshold]

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train[selected_features])
X_test_pca = pca.transform(X_test[selected_features])

model_pca = RandomForestRegressor(max_depth=9, random_state=42)
model_pca.fit(X_train_pca, y_train)

y_train_pred_pca = model_pca.predict(X_train_pca)
y_test_pred_pca = model_pca.predict(X_test_pca)

rmse_train_pca = mean_squared_error(y_train, y_train_pred_pca, squared=False)
rmse_test_pca = mean_squared_error(y_test, y_test_pred_pca, squared=False)

print("----------------------PCA------------------------")
r2_train_pca = r2_score(y_train, y_train_pred_pca)
r2_test_pca = r2_score(y_test, y_test_pred_pca)

print(f"R2 Score with PCA (Train): {r2_train_pca}")
print(f"R2 Score with PCA (Test): {r2_test_pca}")
print(f"RMSE with PCA (Train): {rmse_train_pca}")
print(f"RMSE with PCA (Test): {rmse_test_pca}")



residuals_train_pca = y_train - y_train_pred_pca
residuals_test_pca = y_test - y_test_pred_pca

plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.title("Residual Plot (Train - PCA)")
plt.scatter(y_train_pred_pca, residuals_train_pca, color='blue')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")

plt.subplot(1, 2, 2)
plt.title("Residual Plot (Test - PCA)")
plt.scatter(y_test_pred_pca, residuals_test_pca, color='green')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")

plt.tight_layout()
plt.show()
#-------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------ensamble--------------------------------------------------------------------------
num_selected_features = 4
selected_feature_indices = indices[:num_selected_features]
selected_features = feature_names[selected_feature_indices]

# Create a new DataFrame with the selected top features for the second test
X_test = X[selected_features]

X_train, X_test, y_train, y_test = train_test_split(X_test, y, test_size=0.1, random_state=42)
print("*"*100, "Dataset shapes", "*"*100)
print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_test: {y_test.shape}")
rf_regressor = RandomForestRegressor(max_depth=9, random_state=42)
rf_regressor.fit(X_train, y_train)

y_train_pred = rf_regressor.predict(X_train)
y_test_pred = rf_regressor.predict(X_test)

r2_train_second_test = r2_score(y_train, y_train_pred)
r2_test_second_test = r2_score(y_test, y_test_pred)

rmse_train_second_test = np.sqrt(mean_squared_error(y_train, y_train_pred))
rmse_test_second_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

print("----------------------Features from 1st ensemble------------------------")
print(f"R-squared (R2) for training data: {r2_train_second_test}")
print(f"R-squared (R2) for testing data: {r2_test_second_test}")
print(f"RMSE for training data: {rmse_train_second_test}")
print(f"RMSE for testing data: {rmse_test_second_test}")
print("*"*100)

residuals_train_second_test = y_train - y_train_pred
residuals_test_second_test = y_test - y_test_pred

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_train_pred, residuals_train_second_test)
plt.title("Residuals Plot (Training Data) - Second Test")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")

plt.subplot(1, 2, 2)
plt.scatter(y_test_pred, residuals_test_second_test, color='green')
plt.title("Residuals Plot (Testing Data) - Second Test")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")

plt.show()

#-------------------------------------------------------------------------------------------------------------------------
