import pandas as pd
import scipy.stats as stats
import math

iris_data = pd.read_csv('iris.data')
total_setosa = len(iris_data[iris_data['class'] == 'Iris-setosa'])
total_versicolor = len(iris_data[iris_data['class'] == 'Iris-versicolor'])
total_virginica = len(iris_data[iris_data['class'] == 'Iris-virginica'])

sampled_setosa = iris_data[iris_data["class"] == 'Iris-setosa'].sample(n = 26)
sampled_versicolor = iris_data[iris_data["class"] == 'Iris-versicolor'].sample(n = 27)
sampled_virginica = iris_data[iris_data["class"] == 'Iris-virginica'].sample(n = 27)
iris_data.drop(sampled_setosa.index, inplace=True)
iris_data.drop(sampled_versicolor.index, inplace=True)
iris_data.drop(sampled_virginica.index, inplace=True)

training_set = pd.concat([sampled_virginica, sampled_versicolor, sampled_setosa])

sampled_setosa = iris_data[iris_data["class"] == 'Iris-setosa'].sample(n = 13)
sampled_versicolor = iris_data[iris_data["class"] == 'Iris-versicolor'].sample(n = 12)
sampled_virginica = iris_data[iris_data["class"] == 'Iris-virginica'].sample(n = 13)
iris_data.drop(sampled_setosa.index, inplace=True)
iris_data.drop(sampled_versicolor.index, inplace=True)
iris_data.drop(sampled_virginica.index, inplace=True)

validation_set = pd.concat([sampled_virginica, sampled_versicolor, sampled_setosa])

sampled_setosa = iris_data[iris_data["class"] == 'Iris-setosa'].sample(n = 11)
sampled_versicolor = iris_data[iris_data["class"] == 'Iris-versicolor'].sample(n = 11)
sampled_virginica = iris_data[iris_data["class"] == 'Iris-virginica'].sample(n = 10)
iris_data.drop(sampled_setosa.index, inplace=True)
iris_data.drop(sampled_versicolor.index, inplace=True)
iris_data.drop(sampled_virginica.index, inplace=True)

testing_set = pd.concat([sampled_virginica, sampled_versicolor, sampled_setosa])

training_set.to_csv('train_data_1.csv')
validation_set.to_csv('valid_data_2.csv')
testing_set.to_csv('test_data_3.csv')

