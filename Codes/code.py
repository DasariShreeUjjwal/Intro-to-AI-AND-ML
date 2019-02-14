import numpy as np
import matplotlib.pyplot as plt
import subprocess
import shlex

A = np.array([2*np.sqrt(2)*np.cos(np.pi/12),2*np.sqrt(2)*np.sin(-np.pi/12)])
B = np.array([-2*np.sqrt(2)*np.sin(np.pi/12),2*np.sqrt(2)*np.cos(np.pi/12)])
C = np.array([1-np.sqrt(12),1-np.sqrt(12)])


G = np.array([1,1])
H = np.array([1,-1])
O = np.vstack((G,H))
l = np.zeros(2)
l[0]=2
l[1]=0
Q = np.matmul(np.linalg.inv(O),l)
Y = np.linalg.norm(Q)
print(Q)
print("OQ =" ,np.linalg.norm(Q))
print ("Area =", np.sqrt(3)*3*Y*Y)

len = 10
lam_1 = np.linspace(0,1,len)

x_AB = np.zeros((2,len))
x_BC = np.zeros((2,len))
x_CA = np.zeros((2,len))
for i in range(len):
	temp1 = A + lam_1[i]*(B-A)
	x_AB[:,i]= temp1
	temp2 = B + lam_1[i]*(C-B)
	x_BC[:,i]= temp2
	temp3 = C + lam_1[i]*(A-C)
	x_CA[:,i]= temp3

plt.plot(x_AB[0,:],x_AB[1,:],label='$y + x = 2$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')

plt.plot(A[0],A[1], 'o')
plt.text(A[0]*(1+0.1),A[1]*(1-0.1),'A(2.82,0.012)')
plt.plot(B[0],B[1], 'o')
plt.text(B[0]*(1-0.2),B[1]*(1),'B(-0.012,2.82)')
plt.plot(C[0],C[1], 'o')
plt.text(C[0]*(1+0.03),C[1]*(1-0.1),'C(1-np.sqrt(12),1-np.sqrt(12))')

plt.plot(0,0,'o')
plt.text(0,0+0.1,'$O$')
plt.plot(1,1,'o')
plt.text(1+0.1,1+0.1,'$Q$')

xline=np.linspace(0,1,100)
yline = xline
plt.plot(xline,yline,label='$y - x = 0$')

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')

plt.grid()
plt.axis('equal')
plt.show()
print(temp3)
