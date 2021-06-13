---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.2
  kernelspec:
    display_name: Python [conda env:root] *
    language: python
    name: conda-root-py
---

<h2 style= "text-align: center; color: blue;font-weight: bold;">Wine prediction using Mlflow</h2>


**Author**: Pierre Mulliez \
**Created**: 04/06/2021 \
**Description**: Run a machine learning model using Mlflow \
**Contact**: pierremulliez1@gmail.cim

```python
#Load and import libraries 
import mlflow
import mlflow.sklearn
import pandas as pd 
import numpy as np 
import shutil  
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
```

```python
#change directory 
!cd C:/Users/Pierre Computer/Documents/IE_classes/MLops/Assignment1_Pierre_Mulliez
```

```python
#Delete last runs directory
try:
    shutil.rmtree('./mlruns')
except FileNotFoundError:
    print("WARNING: Can't find folder mlruns")
```

```python
pathdirectory = "./winequality-red.csv"
df = pd.DataFrame(pd.read_csv(pathdirectory))
```

```python
print("The different quality grades in the dataset are: {}".format(np.sort(df.quality.unique())))
print("Do we have any null values ? ")
print({col: np.sum(np.isnan(df.loc[:,col])) for col in df.columns})
df.head()
```

**Train test split**

```python
y = df.quality 
X = df.loc[:,df.columns != "quality"]
```

```python
#Normelize the data
X = preprocessing.normalize(X)
```

```python
#split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=0)
```

## Run the KNN 

```python
#small baseline predict with the mean
y_pred_b = []
for el in range(0,len(y_test)):
    y_pred_b.append(y_train.mean())
```

```python
#Evaluate
print( "Mean abs error: {}".format(mean_absolute_error(y_pred_b, y_test)))
print( "Mean squared error: {}".format(mean_squared_error(y_pred_b, y_test)))
print( "r2 score: {}".format(r2_score(y_pred_b, y_test)))
```

**First model**

```python
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)
y_pred = neigh.predict(X_test)
```

```python
np.unique(y_pred)
```

```python
#Evaluate
mae = mean_absolute_error(y_pred, y_test)
mse = mean_squared_error(y_pred, y_test)
r2 = r2_score(y_pred, y_test)
mape = np.mean(np.abs((y_test - y_pred) / y)) * 100
print( "Mean abs error: {}".format(mae))
print( "Mean squared error: {}".format(mse))
print( "r2 score: {}".format(r2))
```

### Optimization of the model with mlflow


#### Parameters to optimize
- P -> euclidian or mannahatan distance 
- n_neighbors 
- leaf size 

```python
#error function
def calculate_errors(y,ypred):
    mae = mean_absolute_error(y,ypred)
    mse = mean_squared_error(y, ypred)
    R2 = r2_score(y,ypred)
    y,ypred = np.array(y), np.array(ypred)
    mape = np.mean(np.abs((y - ypred) / y)) * 100
    return mae,mse,R2,mape
```

```python
#Mlflow function
def train_knn(nb,leaf,dist,exp = None):
    with mlflow.start_run(experiment_id=exp): #start mlflow run
        neigh = KNeighborsClassifier(n_neighbors=nb, leaf_size= leaf, p = dist)
        neigh.fit(X_train, y_train)
        y_pred = neigh.predict(X_test)
        
        #calculate errors
        mae,mse,R2,mape = calculate_errors(y_test,y_pred)
        errors = mae,mse,R2,mape
        print("MAE:{0:.3f}, MSE:{1:.2f}, R2:{2:.2f}".format(mae, mse, R2))
        
        #log metris and parmeters
        mlflow.log_metrics({"MAE":mae,"MSE":mse, "R2":R2, "MAPE":mape})
        mlflow.log_params({"Numbers of neighbors": nb,"Leaf size":leaf,"p":dist})
        
        #register model
        mlflow.sklearn.log_model(neigh, "model")
        
        #save error plot
        plt.figure()
        plt.bar(['mae','mse','R2','mape'],errors,color=['blue','red','green','orange']);
        plt.title("Errors")
        plt.savefig("errors.png")
        plt.close()
        mlflow.log_artifact("errors.png")
```

```python
exp = mlflow.create_experiment(name="knn normelized")
```

```python
p = [1,2]
nei = [3,4,5,6]
lea = [30,40]
for pdist in p:
    for nb in nei:
        for l in lea:
            train_knn(nb,l,pdist,exp)
```

```python
#log baseline 
mlflow.log_metrics({"MAE":mae,"MSE":mse, "R2":r2, "MAPE":mape})
mlflow.log_params({"Numbers of neighbors": 0,"Leaf size":0,"p":0})
#register model
mlflow.sklearn.log_model("baseline", "model")  
mlflow.end_run()
```

## Best model selector 

```python
df = mlflow.search_runs(experiment_ids="1")
run_id = df.loc[df['metrics.MAE'].idxmin()]['run_id']
print("Minimum error run_id: ",run_id)
```

```python
print('Best number of neighbor: ')
df.loc[df.run_id == run_id,'params.Numbers of neighbors']
```

### As we can see the optimal number of neighbor is 4


# Comparing the run on mlflow 


<img src="files/compare_model.png" alt="Please see screenshot included in the folder called compare model ">



<h2 > Errors with 4 neighbors  </h2>
<img src="files/errors.png" alt="Please see screenshot included in the folder called compare model " >

