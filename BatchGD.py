class MyBatchRegression:
    def __init__(self):
        self.intercept_ = 0.0
        self.coef_ = []

    def fit(self, x, y, learningRate=1, noEpochs=1000):
        self.coef_ = [0.0 for _ in range(len(x[0]) + 1)]
        m = len(x)

        for epochs in range(noEpochs):
            intermediate=[0.0 for _ in range(len(x[0]) + 1)]
            for i in range(len(x)):
                y_computed = self.eval(x[i],self.coef_)
                
                error = (y_computed - y[i])
                
                for j in range(len(x[0])):
                    intermediate[j]+=error*x[i][j]
                    
                intermediate[len(x[0])]+=error
                
            for j in range(len(x[0])):
                self.coef_[j] = self.coef_[j] - learningRate * (1.0 / m) * intermediate[j]
                
            self.coef_[len(x[0])] = self.coef_[len(x[0])] - learningRate * (1.0 / m) * intermediate[len(x[0])]

        self.intercept_ = self.coef_[-1]
        self.coef_ = self.coef_[:-1]

    def eval(self, xi,coef_):
        yi = coef_[-1]

        for j in range(len(xi)):
            yi += coef_[j] * xi[j]

        return yi

    def predict(self, x):
        return [self.eval(xi,self.coef_+[self.intercept_]) for xi in x]
