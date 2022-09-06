########################################
import numpy as np 

class PMM:

    def __init__(self, g, b, AK, BK, A_hat_K, B_hat_K, d1, d2, D, kern, AA_hat_K, AB_hat_K, BA_hat_K, BB_hat_K): #g, b are functions 
        self.g = g
        self.b = b
        self.AK = AK
        self.BK = BK
        self.A_hat_K = A_hat_K
        self.B_hat_K = B_hat_K
        self.d1 = d1  #domain for pde
        self.d2 = d2
        self.D = D  #domain for boundary 
        self.kern = kern 
        self.AA_hat_K = AA_hat_K
        self.AB_hat_K = AB_hat_K
        self.BA_hat_K = BA_hat_K
        self.BB_hat_K = BB_hat_K

    def X_0A(self, mA):
        return [self.d1 + i*(self.d2 - self.d1)/mA for i in range(1, mA)]
    
    def X_0B(self, mB):
        return [self.D[i] for i in range(0, mB)]

    def g_matrix(self, mA):
        return np.vectorize(self.g)(self.X_0A(mA)).T

    def b_matrix(self, mB):
        return np.vectorize(self.b)(self.X_0B(mB)).T

    def Gram(self, X, Y, func):  #X, T are both list 
        output = np.zeros((len(X), len(Y)))
        for i in range(len(X)):
            for j in range(len(Y)):
                output[i-1][j-1] = func(X[i], Y[j]) 
                
        return output 

    def posterior(self, X, mA, mB):
        L_hat_K = np.hstack((self.Gram(X, self.X_0A(mA), self.A_hat_K), self.Gram(X, self.X_0B(mB), self.B_hat_K)))


        LK = np.vstack((self.Gram(self.X_0A(mA), X, self.AK), self.Gram(self.X_0B(mB), X, self.BK))) #LK 


        LL_hat_K = np.vstack((np.hstack((self.Gram(self.X_0A(mA), self.X_0A(mA), self.AA_hat_K), 
                                self.Gram(self.X_0A(mA), self.X_0B(mB), self.AB_hat_K))), 
                                np.hstack((self.Gram(self.X_0B(mB), self.X_0A(mA), self.BA_hat_K), 
                                self.Gram(self.X_0B(mB), self.X_0B(mB), self.BB_hat_K)))))

        latter = np.hstack((self.g_matrix(mA).T, self.b_matrix(mB).T)).T

        middle = np.linalg.pinv(LL_hat_K)
        #middle = np.linalg.inv(LL_hat_K)

        mean = np.matmul(np.matmul(L_hat_K, middle), latter)
        var = self.Gram(X, X, self.kern)  - np.matmul(np.matmul(L_hat_K, middle), LK)

        return mean, var 

