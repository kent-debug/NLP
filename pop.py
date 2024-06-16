# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %%
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

# %%
kent1 = pd.read_excel(r'C:\Users\KENNY PC\Downloads\QUANTIFICATION OF YF campaign for SPT.xlsx ', sheet_name=1, header = 1)

# %%
kent1.head()

# %% [markdown]
# EDA

# %%
kent1.shape

# %%
kent1.describe()

# %% [markdown]
# Finding Missing Values

# %%
kent1.info

# %%
kent1.dtypes

# %%
kent1.isna().sum()

# %%
kent1.value_counts(dropna=True)

# %%
kent1[kent1.isna().any(axis=1)]

# %%
kent1.shape

# %% [markdown]
# We shall remove rows that have missing data through out that is from column 1 to column 26 its empty, from our observation we don't have such a row so nothing was dropped.

# %%
ochom1 = kent1.dropna(how="all")

# %%
ochom1.shape

# %%
ochom1.head()

# %%
ochom1.dropna(subset=['Districts'],inplace=True)

# %%
ochom1.shape

# %%
ochom1.isna().sum()

# %% [markdown]
# Anywhere i have missing refugees, we are going to impute with a value 0.

# %% [markdown]
# Need to add fertility rate
# Mortality rate
# rate of migration

# %% [markdown]
# I will drop columns from my dataset that i believe will not be necessary for our prediction.

# %%
ochom2 = ochom1.drop(['Unnamed: 0 ', 'Nof central supervisors', 'Mother district'], axis=1, errors='ignore')

# %%
ochom2.head()

# %%
ochom2['Refugee population 2022'] = ochom2['Refugee population 2022'].fillna(0)
ochom2['No of municipal councils'] = ochom2['No of municipal councils'].fillna(0)
ochom2['No of islands'] = ochom2['No of islands'].fillna(0)
ochom2['No of hard to reach areas'] = ochom2['No of hard to reach areas'].fillna(0)
ochom2.head(5)

# %%
ochom2 = ochom2.dropna(axis=1, how='all')
ochom2.head()

# %%
ochom2.isna().sum()

# %%
df1 =ochom2.dropna(how="any")
df1.isna().sum()

# %%
df1.head()

# %%
df1.duplicated().sum()

# %%
df1.columns

# %%
df1 = df1.drop(['Hospitals.1', 'HC IV.1', 'HC III.1',
       'HC II.1', 'Clinics.1'],axis=1, errors='ignore')
df1.head()

# %%
df1.shape

# %% [markdown]
# Summary statistics

# %%
df1.describe()

# %%
df1.columns

# %%
columns_to_drop = [
    'No.', ' CONSTITUENCIES  ', '# S/Cs ', '# Parishes', '# LC1s', 
    'No of Town councils', 'No of municipal councils', 'No of city councils', 
    'Total S/Cs', 'total number of PPTs at sub county level', 
    'updated No. Posts to be used', '# of primary schools in 2020', 
    '# of secondary schools IN 2020'
]

# Drop the specified columns
df3 = df1.drop(columns=columns_to_drop, errors='ignore')

# Display the modified dataframe
print(df3.head())

# %%
if all(col in df3.columns for col in ['Hospitals', 'HC IV', 'HC III', 'HC II']):
    # Create the new column by summing the specified columns
    df3['Government Facilities'] = df3[['HC IV', 'HC III', 'HC II']].sum(axis=1)
print(df3)

# %%
columns_to_drop =['Hospitals','HC IV', 'HC III', 'HC II']
df3 = df3.drop(columns=columns_to_drop, errors='ignore')
print(df3.head())

# %%
df3.columns

# %%
sns.set_style("whitegrid")
plt.figure(figsize=(18, 12))
variables = [
    'Distance', 'National Pop 2022', 'National Target propotion- 93%',
    'National Target propotion- under 1 year: 3.58%', 'Refugee population 2022',
    'Refugee Target propotion- 93%', 'Refugee Target propotion- under 1 year: 4.80%',
    'No of islands', 'Clinics', 'Total No of health facilities', 'No of hard to reach areas',
    'Government Facilities'
]

# Create a 4x4 grid to accommodate all variables
fig, axes = plt.subplots(4, 4, figsize=(20, 15))
axes = axes.flatten()  # Flatten the 4x4 array of axes to easily iterate over

for i, var in enumerate(variables):
    sns.histplot(df3[var], kde=True, color='skyblue', ax=axes[i])
    axes[i].set_title(var.replace('_', ' ').title())

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# %%
from sklearn.preprocessing import MinMaxScaler
columns_to_normalize = [
    'Distance', 'National Pop 2022', 'National Target propotion- 93%',
    'National Target propotion- under 1 year: 3.58%', 'Refugee population 2022',
    'Refugee Target propotion- 93%', 'Refugee Target propotion- under 1 year: 4.80%',
    'No of islands', 'Clinics', 'Total No of health facilities', 'No of hard to reach areas',
    'Government Facilities'
]
scaler = MinMaxScaler()
df3[columns_to_normalize] = scaler.fit_transform(df3[columns_to_normalize])

# Plotting
sns.set_style("whitegrid")
plt.figure(figsize=(18, 12))
variables = ['Distance', 'National Pop 2022', 'National Target propotion- 93%',
             'National Target propotion- under 1 year: 3.58%', 'Refugee population 2022',
             'Refugee Target propotion- 93%', 'Refugee Target propotion- under 1 year: 4.80%',
             'No of islands', 'Clinics', 'Total No of health facilities', 'No of hard to reach areas',
             'Government Facilities']

# Create a 3x4 grid to accommodate all variables
fig, axes = plt.subplots(3, 4, figsize=(20, 15))
axes = axes.flatten()  # Flatten the 3x4 array of axes to easily iterate over

for i, var in enumerate(variables):
    sns.histplot(df3[var], kde=True, color='skyblue', ax=axes[i])
    axes[i].set_title(var.replace('_', ' ').title())
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.show()

# %%
df3.head()

# %%
fig, axes = plt.subplots(3, 4, figsize=(20, 15))
axes = axes.flatten()  # Flatten the 3x4 array of axes to easily iterate over

for i, var in enumerate(variables):
    sns.boxplot(y=df3[var], color='skyblue', ax=axes[i])
    axes[i].set_title(var.replace('_', ' ').title())
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# %%
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers
outliers = detect_outliers_iqr(df3, 'National Pop 2022')
print(outliers)


# %% [markdown]
# Dealing with outliers

# %%
def cap_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
    df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
    return df
df_capped = cap_outliers(df3, 'National Pop 2022')


# %%
fig, axes = plt.subplots(3, 4, figsize=(20, 15))
axes = axes.flatten()  # Flatten the 3x4 array of axes to easily iterate over
for i, var in enumerate(variables):
    sns.boxplot(y=df3[var], color='skyblue', ax=axes[i])
    axes[i].set_title(var.replace('_', ' ').title())
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# %%
df3_numeric = df3.select_dtypes(include=[np.number])
correlation_matrix = df3_numeric.corr()
print(correlation_matrix)

# %%
correlation_matrix.columns

# %%
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()

# %% [markdown]
# Model selection

# %%
from sklearn.model_selection import train_test_split
x = correlation_matrix[['Distance', 'National Pop 2022', 'National Target propotion- 93%',
       'National Target propotion- under 1 year: 3.58%',
       'Refugee population 2022', 'Refugee Target propotion- 93%',
       'Refugee Target propotion- under 1 year: 4.80%', 'No of islands',
       'Clinics', 'Total No of health facilities', 'No of hard to reach areas',
       ]] #the predictors
y = correlation_matrix[['Government Facilities']] #the outcome
X_train,X_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=42)

# %%
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

# %%
from sklearn.metrics import mean_squared_error,r2_score ,mean_absolute_error
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)

print(f"Mean Squared error: {mse}")
print(f"R^2 score:{r2}")


# %%
from sklearn.ensemble import RandomForestRegressor , GradientBoostingRegressor

rf_model = RandomForestRegressor()
rf_model.fit(X_train,y_train)
rf_pred = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test,rf_pred)
rf_mae = mean_absolute_error(y_test,rf_pred)
print(rf_mae) 
print(rf_mse)

# %%
gb_model = GradientBoostingRegressor()
gb_model.fit(X_train,y_train)
gb_pred = gb_model.predict(X_test)
gb_mse = mean_squared_error(y_test,gb_pred)
gb_mae = mean_absolute_error(y_test,gb_pred)
print(gb_mae) 
print(gb_mse)

# %%
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

X = correlation_matrix[['Distance', 'National Pop 2022', 'National Target propotion- 93%',
       'National Target propotion- under 1 year: 3.58%',
       'Refugee population 2022', 'Refugee Target propotion- 93%',
       'Refugee Target propotion- under 1 year: 4.80%', 'No of islands',
         'No of hard to reach areas']].values  
y = correlation_matrix[['Government Facilities', 'Clinics', 'Total No of health facilities',]].values  
model = Sequential()
model.add(Dense(64, input_dim=X.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(y.shape[1]))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=50, batch_size=10)
predictions = model.predict(X)

# %%
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# %%
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(y_train.shape[1]))  
model.compile(optimizer='adam', loss='mse')  
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# %%
model.save('my_model.h5')

# %%
features = ['Distance', 'National Pop 2022', 'National Target propotion- 93%',
            'National Target propotion- under 1 year: 3.58%',
            'Refugee population 2022', 'Refugee Target propotion- 93%',
            'Refugee Target propotion- under 1 year: 4.80%', 'No of islands',
            'No of hard to reach areas']
new_data = correlation_matrix[features].iloc[0].values.reshape(1, -1)
new_data = np.array(new_data)
predictions = model.predict(new_data)
print(predictions)


# %% [markdown]
# ********************************************************************************************************************************

# %% [markdown]
# **************************************************************************************************************************************


