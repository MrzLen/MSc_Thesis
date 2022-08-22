import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import norm
from math import sqrt, exp
from mpl_toolkits.mplot3d import Axes3D
 
class BS:
 
    def __init__(self, call, stock, strike, maturity, interest, volatility, dividend):
        self.call = call
        self.stock = stock
        self.strike = strike
        self.maturity = maturity
        self.interest = interest
        self.volatility = volatility
        self.dividend = dividend
        self.d1 = (self.volatility * sqrt(self.maturity)) ** (-1) * (np.log(self.stock / self.strike) + (self.interest-self.dividend + self.volatility ** 2 / 2) * self.maturity)
        self.d2 = self.d1 - self.volatility * sqrt(self.maturity)
 
    def price(self):
        if self.call:
            return exp(-self.dividend * self.maturity) * norm.cdf(self.d1) * self.stock - norm.cdf(self.d2) * self.strike * exp(-self.interest * self.maturity)
        else:
            return norm.cdf(-self.d2) * self.strike * exp(-self.interest * self.maturity) - norm.cdf(-self.d1) * self.stock * exp(-self.dividend * self.maturity)


#Create arrays with the different input values for each variable
from tkinter.tix import Tree


#S = np.linspace(10**(-2) *10, 10**(2) *10, 100) #stock price
#vol = np.linspace(10**(-2) /100, 10**(2) /100, 100) #volatility

S = np.linspace(-2, 2, 100) #stock price
vol = np.linspace(-2, 2, 100) #volatility

 
#Calculate call price for different stock prices and volatility
cs = np.array([])
for i in range(0, len(vol)):
    cs = np.append(cs, BS(True, 10**S *10, 100, 1, 0.05, (10**vol /100)[i], 0).price(), axis=0)
    
cs = cs.reshape(len(S), len(vol))
 
 
#Generate 3D graph for volatility
X2, Y2 = np.meshgrid(S, vol)
 
figcs = plt.figure()
ax = Axes3D(figcs)
ax.plot_surface(X2, Y2, cs, rstride=1, cstride=1, cmap=cm.coolwarm, shade='interp')
#ax.view_init(-140, 60)
plt.title('Call Option Price wrt Volatility')
ax.set_xlabel('Stock Price')
ax.set_ylabel('Volatility')
ax.set_zlabel('Call price')
 
plt.show()