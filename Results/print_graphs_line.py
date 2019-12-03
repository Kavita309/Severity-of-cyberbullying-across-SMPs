# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 00:22:41 2019

@author: hp
"""

import matplotlib.pyplot as plt 
# x-coordinates of left sides of bars  
x = [1, 2, 3, 4] 
  
# heights of bars 
# Twitter char embedding
#y1 = [0.481, 0.408, 0.516, 0.461 ] 
#y2 = [0.252, 0.028, 0.244 , 0.347 ]  
#y3 = [0.185,  0.052, 0.329, 0.393 ]

#Twitter char oversample
#y1 = [0.992, 0.944,0.917, 0.941 ] 
#y2 = [0.992, 0.654 , 0.915,0.992 ]  
#y3 = [0.993,  0.676,0.894, 0.966 ]

#Twitter word embedding
y1 = [0.419 ,0.700, 0.513,0.467 ] 
y2 = [0.286, 0.029,0.294,0.371 ]  
y3 = [0.337,  0.054, 0.370, 0.412 ]

# labels for lines
tick_label = ['Random Forest', 'Naive Bayes', 'LR', 'Svm'] 

plt.xticks(x, tick_label)

# plotting a line chart 
plt.plot(x, y1, color='green',  label = "Precision",linestyle='dashed', linewidth = 3, 
         marker='o', markerfacecolor='black', markersize=7) 

plt.plot(x, y2, color='red', label = "Recall",linestyle='dashed', linewidth = 3, 
         marker='o', markerfacecolor='black', markersize=7) 

plt.plot(x, y3, color='yellow', label = "F1 Score", linestyle='dashed', linewidth = 3, 
         marker='o', markerfacecolor='black', markersize=7)

plt.legend() 
# naming the x-axis 
plt.xlabel('Model-name') 
# naming the y-axis 
plt.ylabel('Performance Matrics for High label') 
# plot title 
plt.title('Twitter word!') 
  
# function to show the plot 
plt.show() 

# =============================================================================
