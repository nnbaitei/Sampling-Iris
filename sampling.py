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

