from sklearn.linear_model import LogisticRegression
import pickle

model = LogisticRegression()

# 模拟一条分界线 y=ax+b, 
X = []
y = []
for i in range(-1000,1000):
    X.append([i])
    if i<=100:
        y.append([1])
    else:
        y.append([2])

model.fit(X, y)

result = model.predict([[4],[200],[-100]])
print('分类结果: ', result)

result = model.predict([[300],[400],[100],[-100]])
print('分类结果: ', result)

modelFilename = 'lesson1_1.p'

pickle.dump(model, open(modelFilename, 'wb'))
 