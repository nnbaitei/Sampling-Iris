import pandas as pd
import scipy.stats as stats
import math

train_data = pd.read_csv('train_data.csv')
valid_data = pd.read_csv('valid_data.csv')
test_data = pd.read_csv('test_data.csv')

# artibutes,Min,Max,Mean,SD
# sepal length,4.3,7.9,5.84,0.83
# sepal width,2.0,4.4,3.05,0.43
# petal length,1.0,6.9,3.76,1.76
# petal width,0.1,2.5,1.20,0.76

mean_sp_length = 5.84
mean_sp_width = 3.05
mean_pt_length = 3.76
mean_pt_width = 1.20
sd_sp_length = 0.83
sd_sp_width = 0.43
sd_pt_length = 1.76
sd_pt_width = 0.76

mean_sp_length_train = sum(train_data['sepal length'])/80
mean_sp_width_train = sum(train_data['sepal width'])/80
mean_pt_length_train = sum(train_data['petal length'])/80
mean_pt_width_train = sum(train_data['petal width'])/80
mean_sp_length_valid = sum(valid_data['sepal length'])/38
mean_sp_width_valid = sum(valid_data['sepal width'])/38
mean_pt_length_valid = sum(valid_data['petal length'])/38
mean_pt_width_valid = sum(valid_data['petal width'])/38
mean_sp_length_test = sum(test_data['sepal length'])/32
mean_sp_width_test = sum(test_data['sepal width'])/32
mean_pt_length_test = sum(test_data['petal length'])/32
mean_pt_width_test = sum(test_data['petal width'])/32

n = 150
sdb_sp_length = sd_sp_length/math.sqrt(n)
sdb_sp_width = sd_sp_width/math.sqrt(n)
sdb_pt_length = sd_pt_length/math.sqrt(n)
sdb_pt_width = sd_pt_width/math.sqrt(n)

z_train_sp_length = (mean_sp_length_train - mean_sp_length)/sdb_sp_length
z_train_sp_width = (mean_sp_width_train - mean_sp_width)/sdb_sp_width
z_train_pt_length = (mean_pt_length_train - mean_pt_length)/sdb_pt_length
z_train_pt_width = (mean_pt_width_train - mean_pt_width)/sdb_pt_width

z_valid_sp_length = (mean_sp_length_valid - mean_sp_length)/sdb_sp_length
z_valid_sp_width = (mean_sp_width_valid - mean_sp_width)/sdb_sp_width
z_valid_pt_length = (mean_pt_length_valid - mean_pt_length)/sdb_pt_length
z_valid_pt_width = (mean_pt_width_valid - mean_pt_width)/sdb_pt_width

z_test_sp_length = (mean_sp_length_test - mean_sp_length)/sdb_sp_length
z_test_sp_width = (mean_sp_width_test - mean_sp_width)/sdb_sp_width
z_test_pt_length = (mean_pt_length_test - mean_pt_length)/sdb_pt_length
z_test_pt_width = (mean_pt_width_test - mean_pt_width)/sdb_pt_width

critical = stats.norm.ppf(0.025)

#Reject H0 or Not Reject H0
if z_train_sp_length < critical:
    result_sp_length_train = "Reject H0"
else:
    result_sp_length_train = "Accept H0"
if z_train_sp_width < critical:
    result_sp_width_train = "Reject H0"
else:
    result_sp_width_train = "Accept H0"
if z_train_pt_length < critical:
    result_pt_length_train = "Reject H0"
else:
    result_pt_length_train = "Accept H0"
if z_train_pt_width < critical:
    result_pt_width_train = "Reject H0"
else:
    result_pt_width_train = "Accept H0"
if z_valid_sp_length < critical:
    result_sp_length_valid = "Reject H0"
else:
    result_sp_length_valid = "Accept H0"
if z_valid_sp_width < critical:
    result_sp_width_valid = "Reject H0"
else:
    result_sp_width_valid = "Accept H0"
if z_valid_pt_length < critical:
    result_pt_length_valid = "Reject H0"
else:
    result_pt_length_valid = "Accept H0"
if z_valid_pt_width < critical:
    result_pt_width_valid = "Reject H0"
else:
    result_pt_width_valid = "Accept H0"
if z_test_sp_length < critical:
    result_sp_length_test = "Reject H0"
else:
    result_sp_length_test = "Accept H0"
if z_test_sp_width < critical:
    result_sp_width_test = "Reject H0"
else:
    result_sp_width_test = "Accept H0"
if z_test_pt_length < critical:
    result_pt_length_test = "Reject H0"
else:
    result_pt_length_test = "Accept H0"
if z_test_pt_width < critical:
    result_pt_width_test = "Reject H0"
else:
    result_pt_width_test = "Accept H0"

data = {
    "sepal length": [mean_sp_length, sd_sp_length, mean_sp_length_train, mean_sp_length_valid, mean_sp_length_test, z_train_sp_length, z_valid_sp_length, z_test_sp_length, critical, result_sp_length_train, result_sp_length_valid, result_sp_length_test],
    "sepal width": [mean_sp_width, sd_sp_length, mean_sp_width_train, mean_sp_width_valid, mean_sp_width_test, z_train_sp_width, z_valid_sp_width, z_test_sp_width, critical, result_sp_width_train, result_sp_width_valid, result_sp_width_test],
    "petal length": [mean_pt_length, sd_pt_length, mean_pt_length_train, mean_pt_length_valid, mean_pt_length_test, z_train_pt_length, z_valid_pt_length, z_test_pt_length, critical, result_pt_length_train, result_pt_length_valid, result_pt_length_test],
    "petal width": [mean_pt_width, sd_pt_width, mean_pt_width_train, mean_pt_width_valid, mean_pt_width_test, z_train_pt_width, z_valid_pt_width, z_test_pt_width, critical, result_pt_width_train, result_pt_width_valid, result_pt_width_test]
}

df = pd.DataFrame(data, index=["Mean", "SD", "Mean of Train set", "Mean of Valid set", "Mean of Test set", "z train", "z valid", "z test", "Z critical", "result(training set)", "result(validation set", "result(testing set)"])

print(df)

