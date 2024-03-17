import pandas as pd
import scipy.stats as stats
import math

iris_data = pd.read_csv('iris.data')
total_setosa = len(iris_data[iris_data['class'] == 'Iris-setosa'])
total_versicolor = len(iris_data[iris_data['class'] == 'Iris-versicolor'])
total_virginica = len(iris_data[iris_data['class'] == 'Iris-virginica'])

training_set = iris_data.sample(n=80)
iris_data.drop(training_set.index, inplace=True)

validation_set = iris_data.sample(n=38)
iris_data.drop(validation_set.index, inplace=True)

testing_set = iris_data.sample(n=32)
iris_data.drop(testing_set.index, inplace=True)

training_set.to_csv('train_data.csv')
validation_set.to_csv('valid_data.csv')
testing_set.to_csv('test_data.csv')

#1.สัดส่วนของชนิดดอกไม้ ทั้ง 3 ชนิด ทั้ง 3 ชุดข้อมูล มีปริมาณเฉลี่ยที่เท่ากัน ในระดับนัยสำคัญ (Significant level) ที่ 0.05
Ho = "proportion of mean in every dataset equal 1/3"
Ha = "proportion of mean in every dataset not equal 1/3"
num_train_vir = len(training_set[training_set['class'] == 'Iris-virginica'])
num_train_set = len(training_set[training_set['class'] == 'Iris-setosa'])
num_train_vers = len(training_set[training_set['class'] == 'Iris-versicolor'])
p_train_vir = num_train_vir/150
p_train_set = num_train_set/150
p_train_vers = num_train_vers/150

num_valid_vir = len(validation_set[validation_set['class'] == 'Iris-virginica'
                                   ])
num_valid_set = len(validation_set[validation_set['class'] == 'Iris-setosa'])
num_valid_vers = len(validation_set[validation_set['class'] == 'Iris-versicolor'])
p_valid_vir = num_valid_vir/150
p_valid_set = num_valid_set/150
p_valid_vers = num_valid_vers/150

num_test_vir = len(testing_set[testing_set['class'] == 'Iris-virginica'])
num_test_set = len(testing_set[testing_set['class'] == 'Iris-setosa'])
num_test_vers = len(testing_set[testing_set['class'] == 'Iris-versicolor'])
p_test_vir = num_test_vir/150
p_test_set = num_test_set/150
p_test_vers = num_test_vers/150

p0_train = (80/3)/150 #training set มี 80 ข้อมูลประกอบด้วยดอกไม้ 3 ชนิด จากข้อมูลดอกไม้ทั้ง 150 ดอก
n = 150
a_train = p0_train*(1-p0_train)/n
p0_valid = (38/3)/150 #validation set มี 38 ข้อมูลประกอบด้วยดอกไม้ 3 ชนิด จากข้อมูลดอกไม้ทั้ง 150 ดอก
a_valid = p0_valid*(1-p0_valid)/n
p0_test = (32/3)/150 #testing set มี 32 ข้อมูลประกอบด้วยดอกไม้ 3 ชนิด จากข้อมูลดอกไม้ทั้ง 150 ดอก
a_test = p0_test*(1-p0_test)/n
p_vir_train = (p_train_vir - p0_train)/math.sqrt(a_train)
p_vers_train = (p_train_vers - p0_train)/math.sqrt(a_train)
p_set_train = (p_train_set - p0_train)/math.sqrt(a_train)
p_vir_valid = (p_valid_vir - p0_valid)/math.sqrt(a_valid)
p_vers_valid = (p_valid_vers - p0_valid)/math.sqrt(a_valid)
p_set_valid = (p_valid_set - p0_valid)/math.sqrt(a_valid)
p_vir_test = (p_test_vir - p0_test)/math.sqrt(a_test)
p_vers_test = (p_test_vers - p0_test)/math.sqrt(a_test)
p_set_test = (p_test_set - p0_test)/math.sqrt(a_test)
critical = stats.norm.ppf(0.025)

#Reject H0 or Not Reject H0
if p_vir_train < critical:
    result_vir_train = "Reject H0"
else:
    result_vir_train = "Accept H0"
if p_vir_valid < critical:
    result_vir_valid = "Reject H0"
else:
    result_vir_valid = "Accept H0"
if p_vir_test < critical:
    result_vir_test = "Reject H0"
else:
    result_vir_test = "Accept H0"
if p_vers_train < critical:
    result_vers_train = "Reject H0"
else:
    result_vers_train = "Accept H0"
if p_vers_valid < critical:
    result_vers_valid = "Reject H0"
else:
    result_vers_valid = "Accept H0"
if p_vers_test < critical:
    result_vers_test = "Reject H0"
else:
    result_vers_test = "Accept H0"
if p_set_train < critical:
    result_set_train = "Reject H0"
else:
    result_set_train = "Accept H0"
if p_set_valid < critical:
    result_set_valid = "Reject H0"
else:
    result_set_valid = "Accept H0"
if p_set_test < critical:
    result_set_test = "Reject H0"
else:
    result_set_test = "Accept H0"

data = {
    "Iris-virginica": [total_virginica, num_train_vir, num_valid_vir, num_test_vir, p_vir_train, p_vir_valid, p_vir_test, critical, result_vir_train, result_vir_valid, result_vir_test],
    "Iris-versicolor": [total_versicolor, num_train_vers, num_valid_vers, num_test_vers, p_vers_train, p_vers_valid, p_vers_test, critical, result_vers_train, result_vers_valid, result_vers_test],
    "Iris-setosa": [total_setosa, num_train_set, num_valid_set, num_test_set, p_set_train, p_set_valid, p_set_test, critical, result_set_train, result_set_valid, result_set_test],
}

df = pd.DataFrame(data, index=["total", "sampling(training set)", "sampling(validation set)", "sampling(testing set)", "z-proportion train", "z-proportion valid", "z-proportion test", "Z critical", "result(training set)", "result(validation set)", "result(testing set)"])

print(df)