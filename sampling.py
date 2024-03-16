import pandas as pd
import scipy.stats as stats

iris_data = pd.read_csv('D:\my_project\Sampling-Iris\iris.data')

training_set = iris_data.sample(n=80)
iris_data.drop(training_set.index, inplace=True)

validation_set = iris_data.sample(n=38)
iris_data.drop(validation_set.index, inplace=True)

testing_set = iris_data.sample(n=32)
iris_data.drop(testing_set.index, inplace=True)


#1.สัดส่วนของชนิดดอกไม้ ทั้ง 3 ชนิด ทั้ง 3 ชุดข้อมูล มีปริมาณเฉลี่ยที่เท่ากัน ในระดับนัยสำคัญ (Significant level) ที่ 0.05
num_train = len(training_set)
num_train_vir = len(training_set[training_set['class'] == 'Iris-virginica'])
num_train_set = len(training_set[training_set['class'] == 'Iris-setosa'])
num_train_vers = len(training_set[training_set['class'] == 'Iris-versicolor'])
mean_train_vir = num_train_vir/num_train
mean_train_set = num_train_set/num_train
mean_train_vers = num_train_vers/num_train
print("mean of virginica from training set", mean_train_vir)
print("mean of setosa from training set", mean_train_set)
print("mean of versicolor from training set", mean_train_vers)

num_valid = len(validation_set)
num_valid_vir = len(validation_set[validation_set['class'] == 'Iris-virginica'])
num_valid_set = len(validation_set[validation_set['class'] == 'Iris-setosa'])
num_valid_vers = len(validation_set[validation_set['class'] == 'Iris-versicolor'])
mean_valid_vir = num_valid_vir/num_valid
mean_valid_set = num_valid_set/num_valid
mean_valid_vers = num_valid_vers/num_valid
print("mean of virginica from validation set", mean_valid_vir)
print("mean of setosa from validation set", mean_valid_set)
print("mean of versicolor from validation set", mean_valid_vers)

num_test = len(testing_set)
num_test_vir = len(testing_set[testing_set['class'] == 'Iris-virginica'])
num_test_set = len(testing_set[testing_set['class'] == 'Iris-setosa'])
num_test_vers = len(testing_set[testing_set['class'] == 'Iris-versicolor'])
mean_test_vir = num_test_vir/num_test
mean_test_set = num_test_set/num_test
mean_test_vers = num_test_vers/num_test
print("mean of virginica from testing set", mean_test_vir)
print("mean of setosa from testing set", mean_test_set)
print("mean of versicolor from testing set", mean_test_vers)

