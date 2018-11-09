import matplotlib.pyplot as plt
import numpy as np
import  pandas as pd
x=np.arange(1,100)
fig=plt.figure()
ax1=fig.add_subplot(331) #2*2的图形 在第一个位置
ax1.plot(x,x)
ax2=fig.add_subplot(332)
ax2.plot(x,-x)
ax3=fig.add_subplot(333)
ax3.plot(x,x**2)
ax3=fig.add_subplot(334)
ax3.plot(x,np.log(x))
plt.show()