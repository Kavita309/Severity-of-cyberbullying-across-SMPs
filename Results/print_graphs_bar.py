# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 12:11:16 2019

@author: hp
"""

import matplotlib.pyplot as plt 
  
# x-coordinates of left sides of bars  
x = [1, 2, 3, 4] 
  
# Twitter embeddings
y = [0.481, 0.419, 0.992, 0.848] 
  
tick_label = ['char', 'word', 'char-oversample', 'word-oversample'] 
  
# plotting a bar chart 
plt.bar(x, y, tick_label = tick_label, 
        width = 0.2, color = ['purple']) 
  
# naming the x-axis 
plt.xlabel('Embedding Type') 
# naming the y-axis 
plt.ylabel('Precision for class High-Random Forest') 
# plot title 
plt.title('Embedding comparison!') 
  
# function to show the plot 
plt.show() 