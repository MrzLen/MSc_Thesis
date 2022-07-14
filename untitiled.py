import numpy as np 

def kern(x1, x2, epsilon = 0.001): #kernel, use Wendland covariance as an example 
        return np.max(1 - np.abs(x1 - x2)/epsilon, 0)**4 * (4*np.abs(x1 - x2)/epsilon + 1)

    
class PMM:
    
    def __init__(self, g, b, A, B, d1, d2, D1, D2): #g, b are functions 
        self.g = g
        self.b = b
        self.A = A
        self.B = B
        self.d1 = d1
        self.d2 = d2
        self.D1 = D1
        self.D2 = D2

    def X_0A(self, mA):
        return [self.d1 + i*(self.d2 - self.d1)/mA for i in range(1, mA)]
    
    def X_0B(self, mB):
        return [self.D1 + i*(self.D2 - self.D1)/mB for i in range(1, mB)]

    def g_matrix(self):
        return np.array([self.X_0A]).T

    def b_matrix(self):
        return np.array([self.X_0B]).T 

    '''
    def A_hat(self):
        return 

    def B_hat(self):
        return 

    def L(self):
        return 

    def L_hat():
        return 
    '''


    def K(X, Y):  #X, T are both list 
        output = np.zeros((len(X), len(Y)))
        for i in range(len(X)):
            for j in range(len(Y)):
                output[i-1][j-1] = kern(X[i], Y[j])

        return output 

    def posterior():
        

