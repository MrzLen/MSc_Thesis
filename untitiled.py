import numpy as np 
import matplotlib.pyplot as plt

def Wendland_kern(x1, x2, epsilon = 0.001): #kernel, use Wendland covariance as an example 
        return np.max(1 - np.abs(x1 - x2)/epsilon, 0)**4 * (4*np.abs(x1 - x2)/epsilon + 1)

def gauss_kernel(x1, x2, L= 0.5):
    return np.exp(-(x1-x2)**2 / (2* L**2))

def gauss_kernel_deri1(x1, x2, L = 0.5):
    return np.exp(-(x1-x2)**2 / (2* L**2)) * (- (x1 - x2) / L**2)

def gauss_kernel_deri2(x1, x2, L = 0.5):
    return np.exp(-(x1-x2)**2 / (2* L**2)) * (1/ L**4) * ((x1-x2)**2  - L**2)

def gauss_kernel_deri2_hat(x1, x2, L = 0.5):
    return gauss_kernel_deri2(x1, x2, L) 

g =  lambda x: np.cos(2*np.pi*x)
b = lambda x: x 

d1, d2 = 0, 1
D1, D2 = 0, 1

class PMM:
    
    def __init__(self, g, b, A, B, A_hat, B_hat, d1, d2, D1, D2, kern): #g, b are functions 
        self.g = g
        self.b = b
        self.A = A
        self.B = B
        self.A_hat = A_hat
        self.B_hat = B_hat
        self.d1 = d1  #domain for pde
        self.d2 = d2
        self.D1 = D1  #domain for boundary 
        self.D2 = D2
        self.kern = kern 

    def X_0A(self, mA):
        return [self.d1 + i*(self.d2 - self.d1)/mA for i in range(1, mA)]
    
    def X_0B(self, mB):
        return [self.D1 + i*(self.D2 - self.D1)/mB for i in range(1, mB)]

    def g_matrix(self, mA):
        return np.array([self.X_0A]).T

    def b_matrix(self):
        return np.array([self.X_0B]).T 


    def K(self, X, Y):  #X, T are both list 
        output = np.zeros((len(X), len(Y)))
        for i in range(len(X)):
            for j in range(len(Y)):
                output[i-1][j-1] = self.kern(X[i], Y[j]) 

        return output 

    def posterior(self, X):
        L_hat_K = np.concatenate(self.A(self.K(X, self.X_0A)), self.B(self.K(X, self.X_0B)))
        L_K = np.concatenate((self.A(self.K(self.X_0A, X)), self.B(self.K(self.X_0B, X))), axis = 1)

        L_L_hat_K = np.bmat([[self.A(self.A_hat(self.K(self.X_0A, self.X_0A))), 
                            self.A(self.B_hat(self.K(self.X_0A, self.X_0B)))], 
                            [self.B(self.A_hat(self.K(self.X_0B, self.X_0A))),
                            self.B(self.B_hat(self.K(self.X_0B, self.X_0A)))]])

        latter = np.concatenate((self.g_matrix, self.b_matrix), axis=1).T

        middle = np.linalg.inv(L_L_hat_K)

        mean = np.matmul(np.matmul(L_hat_K, middle), latter)
        var = self.K(X, X)  - np.matmul(np.matmul(L_hat_K, middle), L_K)

        return mean, var 

    #plots for mean 
    def mean_plot(self):
        plt.plot(np.linspace(self.d1, self.d2), self.mean)





