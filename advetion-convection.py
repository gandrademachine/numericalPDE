#!/usr/bin/env python
# coding: utf-8

# In[78]:


import numpy as np
import matplotlib.pyplot as plt


# In[79]:


# Simulation global data

L = 4.0
alpha = 0.1
t0 = 1.0
tf = 0.0


# In[102]:


# Simulation 1 Data

beta = 0.0;
C = 0.0;
u = 0.0;
s = 1.0;
dx = 0.1;
dt = (s*dx**2)/alpha;
tif = 1.0;
sigma = 0.0;


# In[109]:


# Simulation 2 Data

beta = 0.0;
u = 0.5;
dt = 0.05;
dx = 0.2;
sigma = 0.0;
tif = 1.0;
s = (alpha*dt)/(dx**2);
C = (u*dt)/dx;


# In[123]:


# Simulation 3 Data

beta = 0.0;
u = 4.0;
dt = 0.05;
dx = 0.2;
sigma = 0.0;
tif = 1.0;
s = (alpha*dt)/(dx**2);
C = (u*dt)/dx;


# In[124]:


# Simulation 4 Data

beta = 0.0;
u = 0.5;
dt = 0.05;
dx = 0.2;
sigma = 1.0;
tif = 1.0;
s = (alpha*dt)/(dx**2);
C = (u*dt)/dx;


# In[123]:


# Simulation 5 Data

beta = 0.0;
u = 4.0;
dt = 0.05;
dx = 0.2;
sigma = 1.0;
tif = 1.0;
s = (alpha*dt)/(dx**2);
C = (u*dt)/dx;


# In[123]:


# Simulation 6 Data

beta = 1.0;
u = 1.0;
dt = 0.025;
dx = 0.2;
sigma = 0.0;
tif = 0.5;
s = (alpha*dt)/(dx**2);
C = (u*dt)/dx;


# In[130]:


# Simulation 7 Data

beta = 1.0;
u = 1.0;
dt = 0.025;
dx = 0.2;
sigma = 1.0;
tif = 0.5;
s = (alpha*dt)/(dx**2);
C = (u*dt)/dx;


# In[137]:


# Simulation 8 Data

beta = 0.5;
u = 1.0;
dt = 0.025;
dx = 0.2;
sigma = 0.0;
tif = 0.5;
s = (alpha*dt)/(dx**2);
C = (u*dt)/dx;


# In[139]:


# Simulation 9 Data

beta = 0.5;
u = 1.0;
dt = 0.025;
dx = 0.2;
sigma = 1.0;
tif = 0.5;
s = (alpha*dt)/(dx**2);
C = (u*dt)/dx;


# In[140]:


# Simulation 10 Data

beta = 0.0;
C = 0.125;
s = 0.125;
tmax = 1.0;
sigma = 0.0;
deltx = 0.01;
deltt = 0.05;
u = 0.5;


# In[141]:


# Initialize time and space

x = np.arange(0,L,dx)
n = int(tif/dt)


# In[142]:


# Initial Condition initializing

ci = [1 if i <= 2.0 else 0 for i in (x)]


# In[143]:


# Coefficients
def coef(ci,beta,sigma,C,s):

    B1 = (1.0-beta)*(0.5*C*(1.0+sigma)+s)
    B2 = 1.0-2.0*(1.0-beta)*(0.5*C*sigma+s)
    B3 = (1.0-beta)*(0.5*C*(sigma-1.0)+s)

    a = [1.0+2.0*beta*(0.5*C*sigma+s) for i in (x)]
    b = [beta*(0.5*C*(sigma-1.0)+s) for i in (x)]
    c = [beta*(0.5*C*(1.0+sigma)+s) for i in (x)]
    d = np.zeros(len(x))
    for i in range(len(x)):
        if i == 0: 
            d[i] = c[i]*t0 + B2*ci[i] + B3*ci[i+1]
        elif i == len(x)-1:
            d[i] = B1*ci[i-1] + B2*ci[i] + b[i]*tf
        else: 
            d[i] = B1*ci[i-1] + B2*ci[i] + B3*ci[i+1]
    return a,b,c,d


# In[144]:


# TDMA
def TDMA(ci,beta,sigma,C,s):
 
    P = np.zeros(len(x)-1)
    Q = np.zeros(len(x))
    sol = np.zeros(len(x)+1)
    Q[0] = t0
    a, b, c, d = coef(ci,beta,sigma,C,s)
    for j in range(1,len(x)-1):
        P[j] = b[j]/(a[j]-c[j]*P[j-1])
        Q[j] = (d[j] + c[j]*Q[j-1])/(a[j]-c[j]*P[j-1])
    Q[-1] = tf 
    sol[-1] = Q[-1]
    for j in range(len(x)-2,-1,-1):
        sol[j] = P[j]*sol[j+1] + Q[j] 
    return sol


# In[145]:


def solucao(ci,n,beta,sigma,C,s):
    for t in range(n):
        plt.clf()
        tmp = TDMA(ci,beta,sigma,C,s)
        ci = tmp
        plt.plot(ci, 'bo-')
        plt.pause(1e-3)


# In[146]:

print(ci, n, beta, sigma, C, s,dx,dt)
solucao(ci,n,beta,sigma,C,s)


# In[ ]:





# In[ ]:




