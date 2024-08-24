# Scikit-learn Workflow Revision Notes

## What is Scikit-learn?

Scikit-learn (or sklearn) is a Python library for machine learning. It provides tools for building and evaluating machine learning models and is built on top of NumPy and Matplotlib. Key features include:

- Built-in models for classification, regression, clustering, etc.
- Methods for model evaluation and selection.
- Well-structured APIs for ease of use.

## Machine Learning Workflow

1. **Import Libraries:**
    ```python
    %matplotlib inline
    import matplotlib.pyplot as plt
    import sklearn as sk
    import numpy as np 
    import pandas as pd
    ```

2. **Load Data:**
    ```python
    file = pd.read_csv("../datasets/heart.csv")
    ```

3. **Define Features and Labels:**
    - **Features (X):** All columns except the target.
    - **Labels (y):** The target column.
    ```python
    x = file.drop("target", axis=1)
    y = file["target"]
    ```

4. **Choose and Initialize the Model:**
    ```python
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100)
    ```

5. **Split Data into Training and Testing Sets:**
    ```python
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    ```

6. **Fit the Model:**
    ```python
    clf.fit(x_train, y_train)
    ```

7. **Make Predictions:**
    ```python
    y_preds = clf.predict(x_test)
    ```

8. **Evaluate the Model:**
    ```python
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    
    print(classification_report(y_test, y_preds))
    print(confusion_matrix(y_test, y_preds))
    print("Accuracy:", accuracy_score(y_test, y_preds))
    ```

9. **Improve the Model:**
    - Experiment with different hyperparameters.
    ```python
    np.random.seed(2)
    for i in range(10, 100, 10):
        clf = RandomForestClassifier(n_estimators=i).fit(x_train, y_train)
        print(f"Trying model with {i} estimators...")
        print(f"Model accuracy on the test set: {clf.score(x_test, y_test) * 100:.2f}%")
    ```

10. **Save and Load the Model:**
    ```python
    import pickle
    
    # Save the model
    pickle.dump(clf, open("random_forest.pkl", "wb"))
    print("Saved!!")
    
    # Load the model
    load_model = pickle.load(open("random_forest.pkl", "rb"))
    print("Model accuracy from loaded model:", load_model.score(x_test, y_test))
    ```

## Key Points

- **Features (X):** Variables used to make predictions.
- **Labels (y):** Target variable we want to predict.
- **Model Evaluation:** Use metrics like accuracy, confusion matrix, and classification report.
- **Model Improvement:** Adjust hyperparameters and re-evaluate.

---

## 1. **Data Preparation**

**1.1 Import Libraries**
   ```python
   import pandas as pd
   import numpy as np
   from sklearn.model_selection import train_test_split
   %matplotlib inline
   import matplotlib.pyplot as plt
   ```

**1.2 Load Data**
   ```python
   file = pd.read_csv("../datasets/heart.csv")
   ```

**1.3 Split Data into Features and Labels**
   - **Features (X)**: Everything except the label column.
   - **Labels (Y)**: The column representing the target variable.
   ```python
   x = file.drop("target", axis=1)
   y = file["target"]
   ```

**1.4 Train-Test Split**
   ```python
   x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
   ```

### 2. **Data Cleaning**

**2.1 Identify Missing Values**
   - **Examine Missing Values**: Check for any missing data in the dataset.
   ```python
   file.info()
   ```

**2.2 Remove or Impute Missing Values**
   - **Dropping Missing Values**: Remove rows with missing values.
   ```python
   file = file.dropna()
   ```
   - **Imputing Missing Values**: Fill missing values with a specific strategy (mean, median, mode).
   ```python
   file.fillna(file.mean(), inplace=True)
   ```

**2.3 Transform Data**
   - **Convert Categorical Data to Numerical**: Use encoding techniques like one-hot encoding for categorical variables.
   ```python
   file = pd.get_dummies(file, columns=['Manufacturer', 'Vehicle_type'])
   ```

**2.4 Reduce Data**
   - **Remove Irrelevant Columns**: Drop columns that don't add value to the model.
   ```python
   file = file.drop(["Latest_Launch"], axis=1)
   ```

### 3. **Data Transformation**

**3.1 Feature Engineering**
   - **Create New Features**: Based on existing data, create new features if needed.
   ```python
   file['Engine_to_Weight_Ratio'] = file['Engine_size'] / file['Curb_weight']
   ```

**3.2 Scaling/Normalization**
   - **Standardize Features**: Scale features to ensure they contribute equally.
   ```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   x_scaled = scaler.fit_transform(x)
   ```

### Example Application: Car Sales Data

**1. Load and Clean Data**
   ```python
   file = pd.read_csv("../datasets/Car_sales_missing.csv")
   file = file.drop(["Latest_Launch"], axis=1)
   ```

**2. Handle Missing Values**
   ```python
   file = file.dropna()
   ```

**3. Split Data**
   ```python
   x = file.drop("Price_in_thousands", axis=1)
   y = file["Price_in_thousands"]
   x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
   ```

**4. Encode Categorical Variables**
   ```python
   x = pd.get_dummies(x, columns=['Manufacturer', 'Vehicle_type'])
   ```

**5. Feature Scaling**
   ```python
   scaler = StandardScaler()
   x_scaled = scaler.fit_transform(x)
   ```

**6. Model Training**
   - After preparing and cleaning the data, you can proceed to model training and evaluation.
   ```python
   from sklearn.linear_model import LinearRegression
   model = LinearRegression()
   model.fit(x_train, y_train)
   ```

**7. Model Evaluation**
   ```python
   from sklearn.metrics import mean_squared_error
   y_pred = model.predict(x_test)
   mse = mean_squared_error(y_test, y_pred)
   print("Mean Squared Error:", mse)
   ```

### Summary
- **Import libraries** and **load data**.
- **Split the data** into features and labels, then **train-test split**.
- **Clean the data**: Handle missing values, **transform** categorical variables, and **reduce** unnecessary data.
- **Transform the data**: Feature engineering, scaling, and preparing it for the model.

---

## Data Analysis and Modeling with Python

### 1. Import Libraries

```python
import pandas as pd
import numpy as np
import sklearn 
import matplotlib.pyplot as plt
%matplotlib inline
print("Imported!")
```

- **Libraries Used**: `pandas`, `numpy`, `sklearn`, and `matplotlib` for data handling, machine learning, and plotting.
- **Note**: `%matplotlib inline` is used to display plots directly in Jupyter Notebooks.

### 2. Load and Inspect Data

```python
file = pd.read_csv("../datasets/Car_sales_missing.csv")
file = file.drop("Latest_Launch", axis=1)
file
```

- **Action**: Load the dataset and remove the `Latest_Launch` column.

### 3. Check for Missing Values

```python
file.isna().sum()
```

- **Missing Values Count**:
  - `Manufacturer`: 0
  - `Sales_in_thousands`: 2
  - `__year_resale_value`: 40
  - `Vehicle_type`: 1
  - `Price_in_thousands`: 7
  - `Engine_size`: 4
  - `Horsepower`: 4
  - `Wheelbase`: 4
  - `Width`: 3
  - `Length`: 3
  - `Curb_weight`: 5
  - `Fuel_capacity`: 2
  - `Fuel_efficiency`: 3
  - `Power_perf_factor`: 13

### 4. Handle Missing Values

```python
file["Vehicle_type"].fillna("missing", inplace=True) 
file["__year_resale_value"].fillna(file["__year_resale_value"].mean(), inplace=True)
file["Sales_in_thousands"].fillna(file["Sales_in_thousands"].mean(), inplace=True)
file["Price_in_thousands"].fillna(file["Price_in_thousands"].mean(), inplace=True)
file["Engine_size"].fillna(file["Engine_size"].mean(), inplace=True)
file["Horsepower"].fillna(file["Horsepower"].mean(), inplace=True)
file["Wheelbase"].fillna(file["Wheelbase"].mean(), inplace=True)
file["Width"].fillna(file["Width"].mean(), inplace=True)
file["Length"].fillna(file["Length"].mean(), inplace=True)
file["Curb_weight"].fillna(file["Curb_weight"].mean(), inplace=True)
file["Fuel_capacity"].fillna(file["Fuel_capacity"].mean(), inplace=True)
file["Fuel_efficiency"].fillna(file["Fuel_efficiency"].mean(), inplace=True)
file["Power_perf_factor"].fillna(file["Power_perf_factor"].mean(), inplace=True)
file.isna().sum()
```

- **Action**: Impute missing values:
  - **Categorical Data**: Fill missing values in `Vehicle_type` with "missing".
  - **Numerical Data**: Fill missing values with the mean of respective columns.

### 5. Prepare Data for Modeling

```python
x = file.drop("Price_in_thousands", axis=1)
y = file["Price_in_thousands"]
```

- **Action**: Split the data into features (`x`) and target (`y`).

### 6. One-Hot Encode Categorical Features

```python
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

categorical_features = ["Manufacturer", "Vehicle_type"]
one_hot = OneHotEncoder()
transformer = ColumnTransformer([("one_hot", one_hot, categorical_features)],
                                remainder="passthrough")

transformed_x = transformer.fit_transform(x)
transformed_x
```

- **Action**: Convert categorical variables into numerical format using one-hot encoding.
- **Result**: `transformed_x` is a sparse matrix with 44 features.

### 7. Split Data into Training and Testing Sets

```python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(transformed_x, y, test_size=0.2)
```

- **Action**: Split the data into training and testing sets with 20% of data reserved for testing.

### 8. Train and Evaluate the Model

```python
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()

model.fit(x_train, y_train)
model.score(x_test, y_test)
```

- **Model Used**: `RandomForestRegressor` for predicting `Price_in_thousands`.
- **Model Performance**: R² score of approximately 0.783, indicating the proportion of variance in the target variable explained by the model.

---

## Working with Missing Data and Preparing Data for Modeling Using `sklearn`

### 1. Import Libraries

```python
import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
print("Imported!")
```

- **Libraries Used**: `pandas`, `numpy`, `sklearn`, and `matplotlib` for data manipulation, machine learning, and plotting.
- **Note**: `%matplotlib inline` ensures plots are displayed directly in Jupyter Notebooks.

### 2. Load and Inspect Data

```python
file = pd.read_csv("../datasets/Car_sales_missing.csv")
file["Price_in_thousands"] = file["Price_in_thousands"].fillna(file["Price_in_thousands"].mean())
file.head(20)
```

- **Action**: Load the dataset and impute missing values in `Price_in_thousands` with its mean.

### 3. Inspect Missing Values

```python
x = file.drop("Price_in_thousands", axis=1)
y = file["Price_in_thousands"]
print(x.info())
```

- **Columns with Missing Values**:
  - `Sales_in_thousands`: 2 missing
  - `__year_resale_value`: 40 missing
  - `Engine_size`: 4 missing
  - `Horsepower`: 4 missing
  - `Wheelbase`: 4 missing
  - `Width`: 3 missing
  - `Length`: 3 missing
  - `Curb_weight`: 5 missing
  - `Fuel_capacity`: 2 missing
  - `Fuel_efficiency`: 3 missing
  - `Power_perf_factor`: 13 missing
  - `Latest_Launch`: 5 missing
  - `Manufacturer`, `Vehicle_type`: No missing values

### 4. Impute Missing Values Using `sklearn`

```python
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Impute categorical features with "missing"
category_imputor = SimpleImputer(strategy="constant", fill_value="missing")

# Impute numerical features with mean
num_imputer = SimpleImputer(strategy="mean")

# Define columns
category_features = ["Manufacturer", "Vehicle_type", "Latest_Launch"]
numerical_features = ["Sales_in_thousands", "__year_resale_value", "Engine_size", "Horsepower", "Wheelbase", "Width", "Length", "Curb_weight", "Fuel_capacity", "Fuel_efficiency", "Power_perf_factor"]

# Create ColumnTransformer
imputer = ColumnTransformer([
    ("category_imputor", category_imputor, category_features),
    ("num_imputer", num_imputer, numerical_features)
])

# Transform the data
filled_x = imputer.fit_transform(x)
filled_x
```

- **Action**: Use `SimpleImputer` to handle missing values:
  - **Categorical Features**: Replace missing values with `"missing"`.
  - **Numerical Features**: Replace missing values with the mean.

- **Result**: `filled_x` contains imputed values, ready for further processing.

### 5. Encode Categorical Features

```python
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

categorical_features = ["Manufacturer", "Vehicle_type", "Latest_Launch"]
one_hot = OneHotEncoder()
transformer = ColumnTransformer([
    ("one_hot", one_hot, categorical_features)
], remainder="passthrough")

transformed_x = transformer.fit_transform(filled_x)
transformed_x
```

- **Action**: Apply one-hot encoding to categorical features.
- **Result**: `transformed_x` is a sparse matrix with encoded categorical features and numerical features.

### 6. Split Data into Training and Testing Sets

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(transformed_x, y, test_size=0.2)
```

- **Action**: Split the data into training and testing sets, with 20% reserved for testing.

### 7. Train and Evaluate the Model

```python
model = RandomForestRegressor()
model.fit(x_train, y_train)
model.score(x_test, y_test)
```

- **Model Used**: `RandomForestRegressor` for predicting `Price_in_thousands`.
- **Note**: The model's performance is evaluated using the R² score on the test set.

---


## Notes on Handling Missing Data, Preprocessing, and Model Evaluation using Scikit-learn

### Imports

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
%matplotlib inline

print("Imported!")
```

### 1. Working with Missing Data using Sklearn

#### Loading Data

Load the dataset using pandas and handle missing values:

```python
file = pd.read_csv("../datasets/Car_sales_missing.csv")
```

#### Imputing Missing Data

Use `SimpleImputer` to fill missing values for both categorical and numerical features:

```python
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Define imputers
category_imputer = SimpleImputer(strategy="constant", fill_value="missing")
num_imputer = SimpleImputer(strategy="mean")

# Define columns
category_features = ["Manufacturer", "Vehicle_type", "Latest_Launch"]
numerical_features = ["Sales_in_thousands", "__year_resale_value", "Engine_size", "Horsepower", "Wheelbase", "Width", "Length", "Curb_weight", "Fuel_capacity", "Fuel_efficiency", "Power_perf_factor"]

# Create ColumnTransformer
imputer = ColumnTransformer([
    ("category_imputor", category_imputer, category_features),
    ("num_imputer", num_imputer, numerical_features)
])

# Transform data
filled_x = imputer.fit_transform(x)
```

#### Encoding Categorical Data

Use `OneHotEncoder` for categorical data and combine it with numerical data:

```python
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

categorical_features = ["Manufacturer", "Vehicle_type", "Latest_Launch"]
one_hot = OneHotEncoder()

# Transform data
transformer = ColumnTransformer([
    ("one_hot", one_hot, categorical_features)
], remainder="passthrough")

transformed_x = transformer.fit_transform(x)
```

#### Model Training and Evaluation

Use `RandomForestRegressor` for regression:

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Split data
x_train, x_test, y_train, y_test = train_test_split(transformed_x, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor()
model.fit(x_train, y_train)

# Evaluate model
score = model.score(x_test, y_test)
print("RandomForestRegressor score:", score)
```

### 2. Working with Built-in Dataset (Diabetes Dataset)

#### Load the Dataset

Load and inspect the diabetes dataset:

```python
from sklearn.datasets import load_diabetes

# Load dataset
f = load_diabetes()
file = pd.DataFrame(f["data"], columns=f["feature_names"])
file["target"] = pd.Series(f["target"])

print(file.info())
```

#### Splitting Data

Prepare data for training:

```python
# Prepare features and target
x = file.drop("target", axis=1)
y = file["target"]

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```

#### Model Training and Evaluation

Use Ridge Regression:

```python
from sklearn.linear_model import Ridge

# Train model
model = Ridge()
model.fit(x_train, y_train)

# Evaluate model
score = model.score(x_test, y_test)
print("Ridge Regression score:", score)
```

Compare with Random Forest Regressor:

```python
from sklearn.ensemble import RandomForestRegressor

# Train model
rf = RandomForestRegressor()
rf.fit(x_train, y_train)

# Evaluate model
rf_score = rf.score(x_test, y_test)
print("RandomForestRegressor score:", rf_score)
```

### 3. Classification Example

#### Classification with Built-in Dataset

Use `LinearSVC`, `NaiveBayes`, and `RandomForestClassifier` for classification:

```python
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()
x = pd.DataFrame(iris["data"], columns=iris["feature_names"])
y = pd.Series(iris["target"])

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train and evaluate models
svc = LinearSVC()
svc.fit(x_train, y_train)
print("LinearSVC score:", svc.score(x_test, y_test))

nb = GaussianNB()
nb.fit(x_train, y_train)
print("Naive Bayes score:", nb.score(x_test, y_test))

rf = RandomForestClassifier()
rf.fit(x_train, y_train)
print("RandomForestClassifier score:", rf.score(x_test, y_test))
```

### Summary

- **Missing Data Handling**: Use `SimpleImputer` for filling missing values. For categorical data, use `OneHotEncoder`.
- **Regression Models**: Evaluate different models such as `Ridge` and `RandomForestRegressor`.
- **Classification Models**: Compare models such as `LinearSVC`, `Naive Bayes`, and `RandomForestClassifier`.




