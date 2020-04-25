
#this file is an numerical simulation

import numpy as np
import matplotlib.pyplot as plt

#non flux limiter schemes
def upwind(array,index,dx):
    return 0
def Lax(array,index,dx):
    return (array[index+1]-array[index])/dx
def Fromm(array,index,dx):
   return (array[index+1]-array[index-1])/(2*dx)
def BeamWarming(array,index,dx):
    return (array[index]-array[index-1])/dx
#flux limiter schemes
def Koren(array,index,dx):
    t=(array[index-1]-array[index-2])/(array[index+1]-array[index])
    return max(0,min(2*t,min((1+2*t)/3,2)))
def Ospre(array,index,dx):
    t=(array[index-1]-array[index-2])/(array[index+1]-array[index]+np.exp(-12))
    return 1.5*(t**2+t)/(t**2+t+1)
def HCUS(array,index,dx):
    t=(array[index-1]-array[index-2])/(array[index+1]-array[index]+np.exp(-12))
    return 1.5*(t+abs(t))/(t+2)
def HQUICK(array,index,dx):
    t=(array[index-1]-array[index-2])/(array[index+1]-array[index]+np.exp(-12))
    return 2*(t+abs(t))/(t+3)
def Minmod(array,index,dx):
    t=(array[index-1]-array[index-2])/(array[index+1]-array[index]+np.exp(-12))
    return max(0,min(1,t))
def MC(array,index,dx):
    t=(array[index-1]-array[index-2])/(array[index+1]-array[index]+np.exp(-12))
    return max(0,min(2*t,0.5*(1+t),2))
def UMIST(array,index,dx):
    t=(array[index-1]-array[index-2])/(array[index+1]-array[index]+np.exp(-12))
    return max(0,min(2*t,0.5*(0.5+1.5*t),0.5*(1.5+0.5*t),2))
def Superbee(array,index,dx):
    t=(array[index-1]-array[index-2])/(array[index+1]-array[index]+np.exp(-12))
    return max(0,min(2*t,1),min(t,2))
def Sweby(array,index,dx):
    b = 0.5
    t=(array[index-1]-array[index-2])/(array[index+1]-array[index]+np.exp(-12))
    return max(0,min(b*t,1),min(t,2))
def Smart(array,index,dx):
    t=(array[index-1]-array[index-2])/(array[index+1]-array[index]+np.exp(-12))
    return max(0,min(2*t,0.5*(0.5+1.5*t),4))
def VanAlbada1(array,index,dx):
    t=(array[index-1]-array[index-2])/(array[index+1]-array[index]+np.exp(-12))
    return (t**2+t)/(t**2+1)
def VanAlbada2(array,index,dx):
    t=(array[index-1]-array[index-2])/(array[index+1]-array[index]+np.exp(-12))
    return t*2/(t**2+1)
def VanLeer(array,index,dx):
    t=(array[index-1]-array[index-2])/(array[index+1]-array[index]+np.exp(-12))
    return (t+abs(t))/(abs(t)+1)
#defining slope function
def slope(array,index,dx,method):
    if method == 'upwind':
        return upwind(array,index,dx)
    elif method == 'LaxWendroff':
        return Lax(array,index,dx)
    elif method =='BeamWarming':
        return BeamWarming(array,index,dx)
    elif method == 'Koren':
        return Koren(array,index,dx)
    elif method == 'Ospre':
        return Ospre(array,index,dx)
    elif method == 'HCUS':
        return HCUS(array,index,dx)
    elif method == 'HQUICK':
        return HQUICK(array,index,dx)
    elif method == 'Minmod':
        return Minmod(array,index,dx)
    elif method == 'Monotonized':
        return MC(array,index,dx)
    elif method == 'Smart':
        return Smart(array,index,dx)
    elif method == 'Superbee':
        return Superbee(array,index,dx)
    elif method == 'Sweby':
        return Sweby(array,index,dx)
    elif method == 'UMIST':
        return UMIST(array,index,dx)
    elif method == 'VanAlbada1':
        return VanAlbada1(array,index,dx)
    elif method == 'VanAlbada2':
        return VanAlbada2(array,index,dx)
    elif method == 'VanLeer':
        return VanLeer(array,index,dx)
    else:
         return 0
def Godunov(u,dx,dt,tf,method):
    x = np.arange(0-dx,1.0+dx,dx)
    ICArray = [np.exp(-200*(i-0.3)**2)+1 if(i>=0.6 and i<=0.8) else np.exp(-200*(i-0.3)**2) for i in x]
    ICArray[0] = ICArray[-2]
    ICArray[-1] = ICArray[1]
    t0 = 0
    sstp = int(len(ICArray))
    tstp = int(tf/dt)
    for i in range(tstp):
        temp = np.zeros(sstp)
        ICArray[0] = ICArray[-2]
        ICArray[-1] = ICArray[1]
        plt.clf()
        for j in range(1,sstp-1):
             temp[j] =  ICArray[j]-(u*dt*(ICArray[j]-ICArray[j-1])+0.5*(u*dx*dt-(u*dt)**2)*(slope(ICArray,j,dx,method)-slope(ICArray,j-1,dx,method)))/dx
        ICArray = temp
        t0 +=dt
        plt.plot(x,ICArray,'bo-')
        plt.suptitle("t = %1.3f" %(t0))
        plt.grid(True)
        plt.axis([0,1,-1,2])
        plt.pause(0.001)
#nonlinear methods
def viscose(array,index,dx,dt,mi):
    return dt*mi*(array[index+1]-array[index])/dx**2
def nonviscose(array,index,dx,dt,mi):
    alpha = max(abs(array[index+1]),abs(array[index]))
    return 0.5*(dt/dx)*alpha*(array[index+1]-array[index])
def nonlinear_slope(array,index,dx,dt,mi,method):
    if method ==  'viscose':
        return viscose(array,index,dx,dt,mi)
    elif method == 'nonviscose':
        return nonviscose(array,index,dx,dt,mi)
    else:
        return 0
def Godunov_nonlinear(mi,dx,dt,tf,method):
    x = np.arange(0-dx,1.0+dx,dx)
    ICArray = [np.exp(-200*(i-0.3)**2) for i in x]
    ICArray[0] = ICArray[-2]
    ICArray[-1] = ICArray[1]
    t0 = 0
    sstp = int(len(ICArray))
    tstp = int(tf/dt)
    for i in range(tstp):
        temp = np.zeros(sstp)
        ICArray[0] = ICArray[-2]
        ICArray[-1] = ICArray[1]
        plt.clf()
        for j in range(1,sstp-1):
             temp[j] =  ICArray[j]-0.25*dt*(ICArray[j+1]**2-ICArray[j-1]**2)/dx+(nonlinear_slope(ICArray,j,dx,dt,mi,method)-nonlinear_slope(ICArray,j-1,dx,dt,mi,method))
        ICArray = temp
        t0 +=dt
        plt.plot(x,ICArray,'bo-')
        plt.suptitle("t = %1.3f" %(t0))
        plt.grid(True)
        plt.axis([0,1,-1,2])
        plt.pause(0.001)
# Godunov(u=1.0,dx=0.01,dt=0.008,tf=2,method='VanLeer')    
Godunov_nonlinear(mi=1.0,dx=0.01,dt=0.01,tf=2,method='viscose')  
