# Homework 1: Introduction & Prerequisites

**Homework submission link**: https://courses.datatalks.club/mlops-zoomcamp-2024/homework/hw1

**Deadline**: 20 May 2024

## Machine Learning

In this homework we will use the [NYC taxi dataset](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page), but instead of "**Green** Taxi Trip Records", we will use "**Yellow** Taxi Trip Records".

### Question 1: Downloading the data

Download the data for January and February 2023.

Read the data for January. How many columns are there?

- 16
- 17
- 18
- **19**    <-- answer

**Answer:**

```python
# read data
train = pd.read_parquet("data/yellow_tripdata_2023-01.parquet")
valid = pd.read_parquet("data/yellow_tripdata_2023-02.parquet")

# number of columns in January 2023
print (f"Number of columns in January 2023 Yellow Taxi trip records = {len(train.columns)}")
```

### Question 2: Computing duration

Now let's compute the `duration` variable. It should contain the duration of a ride in minutes. 

What's the standard deviation of the trips duration in January?

- 32.59
- **42.59**    <-- answer
- 52.59
- 62.59

**Answer:**

```python
# compute trip duration
train['duration'] = train["tpep_dropoff_datetime"] - train["tpep_pickup_datetime"]
train['duration'] = train['duration'].apply(lambda td: td.total_seconds() / 60)

# standard deviation of the trips duration in January 2023
duration_std = round(train['duration'].std(), 2)
print (f"The standard deviation of the trip duration (minutes) in January 2023 = {duration_std}")
```

### Question 3: Dropping outliers

Next, we need to check the distribution of the `duration` variable. There are some outliers. Let's remove them and keep only the records where the duration was between 1 and 60 minutes (inclusive).

What fraction of the records are left after you dropped the outliers?

- 90%
- 92%
- 95%
- **98%**    <-- answer

**Answer**:

```python
# remove outliers and keep only the records where the duration was between 1 and 60 minutes (inclusive)
train_wo_outliers = train[(train["duration"]>=1)&(train["duration"]<=60)]

# fraction of the records left after dropping the outliers
fraction = round(len(train_wo_outliers)*100/len(train))
print (f"The fraction of the records left after dropping the outliers = {fraction}%")
```

### Question 4: One-hot encoding

Let's apply one-hot encoding to the pickup and dropoff location IDs. We'll use only these two features for our model. 

- Turn the dataframe into a list of dictionaries (remember to re-cast the ids to strings - otherwise it will label encode them)
- Fit a dictionary vectorizer 
- Get a feature matrix from it

What's the dimensionality of this matrix (number of columns)?

- 2
- 155
- 345
- **515**    <-- answer
- 715

**Answer**:

```python
# apply one-hot encoding to the pickup and dropoff location IDs
categorical = ["PULocationID", "DOLocationID"]
train_wo_outliers[categorical] = train_wo_outliers[categorical].astype(str)
train_dicts = train_wo_outliers[categorical].to_dict(orient="records")
dv = DictVectorizer()
X_train = dv.fit_transform(train_dicts)

# number of columns in the feature matrix
print (f"Number of columns in the feature matrix = {X_train.shape[1]}")
```

### Question 5: Training a model

Now let's use the feature matrix from the previous step to train a model. 

- Train a plain linear regression model with default parameters 
- Calculate the RMSE of the model on the training data

What's the RMSE on train?

- 3.64
- **7.64**    <-- answer
- 11.64
- 16.64

**Answer**:

```python
# train a linear regression model with default parameters
target = "duration"
y_train = train_wo_outliers[target].values

lr = LinearRegression()
lr.fit(X_train, y_train)

# calculate the RMSE of the model on the training data
train_rmse = round(root_mean_squared_error(y_train, lr.predict(X_train)), 2)
print (f"RMSE on train data = {train_rmse}")
```

### Question 6: Evaluating the model

Now let's apply this model to the validation dataset (February 2023). 

What's the RMSE on validation?

- 3.81
- **7.81**    <-- answer
- 11.81
- 16.81

**Answer**:

```python
# apply model to validation data
valid['duration'] = valid["tpep_dropoff_datetime"] - valid["tpep_pickup_datetime"]
valid['duration'] = valid['duration'].apply(lambda td: td.total_seconds() / 60)
valid = valid[(valid["duration"]>=1)&(valid["duration"]<=60)]
valid[categorical] = valid[categorical].astype(str)
valid_dicts = valid[categorical].to_dict(orient="records")
X_valid = dv.transform(valid_dicts)
y_valid = valid[target].values
print (f"Number of columns in the feature matrix = {X_valid.shape[1]}")

# calculate the RMSE of the model on the validation data
valid_rmse = round(root_mean_squared_error(y_valid, lr.predict(X_valid)), 2)
print (f"RMSE on validation data = {valid_rmse}")
```