import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from prettytable import PrettyTable


def func(model, days):
    return model.predict(polyFeat.fit_transform([[days]]))


### LOAD DATA ###
data = pd.read_csv('10CountriesCovidData.csv', sep=',')
data = data[
    ['id', 'Afghanistan', 'Brazil', 'Finland', 'Germany', 'India', 'Japan', 'Italy', 'Kenya', 'Malaysia', 'Nepal',
     'Ukraine']]
print('-' * 30);
print("HEAD");
print('-' * 30)
print(data.head())

###PREPARE DATA ###
# print('-'*30);print("PREPARE DATA");print('-'*30)
x = np.array(data['id']).reshape(-1, 1)
y0 = np.array(data['Afghanistan']).reshape(-1, 1)
y1 = np.array(data['Brazil']).reshape(-1, 1)
y2 = np.array(data['Finland']).reshape(-1, 1)
y3 = np.array(data['Germany']).reshape(-1, 1)
y4 = np.array(data['India']).reshape(-1, 1)
y5 = np.array(data['Japan']).reshape(-1, 1)
y6 = np.array(data['Italy']).reshape(-1, 1)
y7 = np.array(data['Kenya']).reshape(-1, 1)
y8 = np.array(data['Malaysia']).reshape(-1, 1)
y9 = np.array(data['Nepal']).reshape(-1, 1)
y10 = np.array(data['Ukraine']).reshape(-1, 1)
# plt.plot(y,'-m')
# plt.show()

polyFeat = PolynomialFeatures(degree=8)
x = polyFeat.fit_transform(x)
# print(x)

###TRAINING DATA###
# print('-'*30);print("TRAINING DATA");print('-'*30)
model0 = linear_model.LinearRegression()
model0.fit(x, y0)
model1 = linear_model.LinearRegression()
model1.fit(x, y1)
model2 = linear_model.LinearRegression()
model2.fit(x, y2)
model3 = linear_model.LinearRegression()
model3.fit(x, y3)
model4 = linear_model.LinearRegression()
model4.fit(x, y4)
model5 = linear_model.LinearRegression()
model5.fit(x, y5)
model6 = linear_model.LinearRegression()
model6.fit(x, y6)
model7 = linear_model.LinearRegression()
model7.fit(x, y7)
model8 = linear_model.LinearRegression()
model8.fit(x, y8)
model9 = linear_model.LinearRegression()
model9.fit(x, y9)
model10 = linear_model.LinearRegression()
model10.fit(x, y10)
# accuracy = model.score(x,y)
# print(f'Accuracy: {round(accuracy*100,2)}')
# y0=model.predict(x)
# plt.plot(y0,'--b')
# plt.show()

###PREDICTION###
daysCount = 1048
days = 10
print('-' * 30);
print("PREDICTION");
print('-' * 30)
print("Cases in the tenth day from 04/12/2022")
print('Afghanistan - ', int(model0.predict(polyFeat.fit_transform([[daysCount + days]]))))
print('Brazil      - ', int(model1.predict(polyFeat.fit_transform([[daysCount + days]]))))
print('Finland     - ', int(model2.predict(polyFeat.fit_transform([[daysCount + days]]))))
print('Germany     - ', int(model3.predict(polyFeat.fit_transform([[daysCount + days]]))))
print('India       - ', int(model4.predict(polyFeat.fit_transform([[daysCount + days]]))))
print('Japan       - ', int(model5.predict(polyFeat.fit_transform([[daysCount + days]]))))
print('Italy       - ', int(model6.predict(polyFeat.fit_transform([[daysCount + days]]))))
print('Kenya       - ', int(model7.predict(polyFeat.fit_transform([[daysCount + days]]))))
print('Malaysia    - ', int(model8.predict(polyFeat.fit_transform([[daysCount + days]]))))
print('Nepal       - ', int(model9.predict(polyFeat.fit_transform([[daysCount + days]]))))
print('Ukraine     - ', int(model10.predict(polyFeat.fit_transform([[daysCount + days]]))))
print('-' * 30);
print("PREDICTION IN THE FOLLOWING 10 DAYS");
print('-' * 30)
print("Cases in the next ten days from 04/12/2022")
tb = PrettyTable(["COUNTRIES","05/12/2022","06/12/2022","07/12/2022","08/12/2022","09/12/2022","10/12/2022","11/12/2022","12/12/2022","13/12/2022","14/12/2022"])
# print("               05/12/2022  06/12/2022  07/12/2022  08/12/2022  09/12/2022  10/12/2022  11/12/2022 12/12/2022  13/12/2022 14/12/2022")
# print('Afghanistan - ',' ',int(func(model0,daysCount+1)),int(func(model0,daysCount+2)),int(func(model0,daysCount+3)),int(func(model0,daysCount+4)),int(func(model0,daysCount+5)),int(func(model0,daysCount+6)),int(func(model0,daysCount+7)),int(func(model0,daysCount+8)),int(func(model0,daysCount+9)),int(func(model0,daysCount+10)))
# print('Brazil      - ',int(model1.predict(polyFeat.fit_transform([[1049]]))))
# print('Finland     - ',int(model2.predict(polyFeat.fit_transform([[1049]]))))
# print('Germany     - ',int(model3.predict(polyFeat.fit_transform([[1049]]))))
# print('India       - ',int(model4.predict(polyFeat.fit_transform([[1049]]))))
# print('Japan       - ',int(model5.predict(polyFeat.fit_transform([[1049]]))))
# print('Italy       - ',int(model6.predict(polyFeat.fit_transform([[1049]]))))
# print('Kenya       - ',int(model7.predict(polyFeat.fit_transform([[1049]]))))
# print('Malaysia    - ',int(model8.predict(polyFeat.fit_transform([[1049]]))))
# print('Nepal       - ',int(model9.predict(polyFeat.fit_transform([[1049]]))))
# print('Ukraine     - ',int(model10.predict(polyFeat.fit_transform([[1049]]))))
tb.add_row(["Afghanistan",int(func(model0,daysCount+1)),int(func(model0,daysCount+2)),int(func(model0,daysCount+3)),int(func(model0,daysCount+4)),int(func(model0,daysCount+5)),int(func(model0,daysCount+6)),int(func(model0,daysCount+7)),int(func(model0,daysCount+8)),int(func(model0,daysCount+9)),int(func(model0,daysCount+10))])
tb.add_row(["Brazil",int(func(model1,daysCount+1)),int(func(model1,daysCount+2)),int(func(model1,daysCount+3)),int(func(model1,daysCount+4)),int(func(model1,daysCount+5)),int(func(model1,daysCount+6)),int(func(model1,daysCount+7)),int(func(model1,daysCount+8)),int(func(model1,daysCount+9)),int(func(model1,daysCount+10))])
tb.add_row(["Finland",int(func(model2,daysCount+1)),int(func(model2,daysCount+2)),int(func(model2,daysCount+3)),int(func(model2,daysCount+4)),int(func(model2,daysCount+5)),int(func(model2,daysCount+6)),int(func(model2,daysCount+7)),int(func(model2,daysCount+8)),int(func(model2,daysCount+9)),int(func(model2,daysCount+10))])
tb.add_row(["Germany",int(func(model3,daysCount+1)),int(func(model3,daysCount+2)),int(func(model3,daysCount+3)),int(func(model3,daysCount+4)),int(func(model3,daysCount+5)),int(func(model3,daysCount+6)),int(func(model3,daysCount+7)),int(func(model3,daysCount+8)),int(func(model3,daysCount+9)),int(func(model3,daysCount+10))])
tb.add_row(["India",int(func(model4,daysCount+1)),int(func(model4,daysCount+2)),int(func(model4,daysCount+3)),int(func(model4,daysCount+4)),int(func(model4,daysCount+5)),int(func(model4,daysCount+6)),int(func(model4,daysCount+7)),int(func(model4,daysCount+8)),int(func(model4,daysCount+9)),int(func(model4,daysCount+10))])
tb.add_row(["Japan",int(func(model5,daysCount+1)),int(func(model5,daysCount+2)),int(func(model5,daysCount+3)),int(func(model5,daysCount+4)),int(func(model5,daysCount+5)),int(func(model5,daysCount+6)),int(func(model5,daysCount+7)),int(func(model5,daysCount+8)),int(func(model5,daysCount+9)),int(func(model5,daysCount+10))])
tb.add_row(["Italy",int(func(model6,daysCount+1)),int(func(model6,daysCount+2)),int(func(model6,daysCount+3)),int(func(model6,daysCount+4)),int(func(model6,daysCount+5)),int(func(model6,daysCount+6)),int(func(model6,daysCount+7)),int(func(model6,daysCount+8)),int(func(model6,daysCount+9)),int(func(model6,daysCount+10))])
tb.add_row(["Kenya",int(func(model7,daysCount+1)),int(func(model7,daysCount+2)),int(func(model7,daysCount+3)),int(func(model7,daysCount+4)),int(func(model7,daysCount+5)),int(func(model7,daysCount+6)),int(func(model7,daysCount+7)),int(func(model7,daysCount+8)),int(func(model7,daysCount+9)),int(func(model7,daysCount+10))])
tb.add_row(["Malaysia",int(func(model8,daysCount+1)),int(func(model8,daysCount+2)),int(func(model8,daysCount+3)),int(func(model8,daysCount+4)),int(func(model8,daysCount+5)),int(func(model8,daysCount+6)),int(func(model8,daysCount+7)),int(func(model8,daysCount+8)),int(func(model8,daysCount+9)),int(func(model8,daysCount+10))])
tb.add_row(["Nepal",int(func(model9,daysCount+1)),int(func(model9,daysCount+2)),int(func(model9,daysCount+3)),int(func(model9,daysCount+4)),int(func(model9,daysCount+5)),int(func(model9,daysCount+6)),int(func(model9,daysCount+7)),int(func(model9,daysCount+8)),int(func(model9,daysCount+9)),int(func(model9,daysCount+10))])
tb.add_row(["Ukraine",int(func(model10,daysCount+1)),int(func(model10,daysCount+2)),int(func(model10,daysCount+3)),int(func(model10,daysCount+4)),int(func(model10,daysCount+5)),int(func(model10,daysCount+6)),int(func(model10,daysCount+7)),int(func(model10,daysCount+8)),int(func(model10,daysCount+9)),int(func(model10,daysCount+10))])

# x1 = np.array(list(range(1,1048+days))).reshape(-1,1)
# y1 = model.predict(polyFeat.fit_transform(x1))
#
# plt.plot(y1,'--r')
# plt.show()
print(tb)