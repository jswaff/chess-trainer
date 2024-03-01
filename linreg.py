import numpy as np
import pandas as pd

class LinearRegressionGD:

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.array([0.])
        self.losses_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * 2.0 * errors.mean()
            loss = (errors**2).mean()
            self.losses_.append(loss)
        return self

    def net_input(self, X):
        return np.dot(X,self.w_) + self.b_

    def predict(self, X):
        return self.net_input(X)


df = pd.read_csv('data/mat0.csv', header=None, nrows=1024*10)
X = df.iloc[:, 0:10].values
y = df.iloc[:, 10].values
print(f'X: {X.shape} {X}')
print(f'y: {y.shape} {y}')

# from sklearn.preprocessing import StandardScaler
# sc_x = StandardScaler()
# sc_y = StandardScaler()
# X_std = sc_x.fit_transform(X)
# y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()
# print(f'X_std: {X_std.shape} {X_std}')
# print(f'y_std: {y_std.shape} {y_std}')

# lr = LinearRegressionGD(eta=0.001, n_iter=10000)
# lr.fit(X, y)
# #
# print(f'weights: {lr.w_}')
# print(f'bias: {lr.b_}')
# #
# y_pred = lr.predict(X)
# #y_pred_reverted = sc_y.inverse_transform(y_pred.reshape(-1,1))
# for y_t, y_p in zip(y[:10], y_pred):
#     print(f'y: {y_t} y_pred: {y_p}')
#


from sklearn.linear_model import LinearRegression
slr = LinearRegression()
slr.fit(X, y)
y_pred = slr.predict(X)
#y_pred_reverted = sc_y.inverse_transform(y_pred.reshape(-1,1))
print(f'Coefficients: {slr.coef_}:.3f')
print(f'Intercept: {slr.intercept_:.3f}')
for y_t, y_p in zip(y[:10], y_pred):
   print(f'y: {y_t} y_pred: {y_p}')

#import matplotlib.pyplot as plt
#plt.plot(range(1, lr.n_iter+1), lr.losses_)
#plt.ylabel('MSE')
#plt.xlabel('Epoch')
#plt.show()

