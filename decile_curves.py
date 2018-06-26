import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#a function which takes in an array of predicted values and returns the percentile associated with each one
def decile_gen(arr_y_pred):
	return np.array(pd.qcut(pd.Series(arr_y_pred).rank(method='first'),10,labels=range(10,0,-1)))  #method = first is used in the case when there are a lot of 0s and overlapping of labels

#decile plot takes in 2 numpy arrays as arguments. the first one is the actual target values from the 
#test sample and the second is the predicted probabilities in the test sample
#the program then ranks the samples by their predicted score and breaks them into deciles
#afterwards it calculates the average predicted probability per decide and the average for the actuals
#per decile and plots the results
def decile_plot(arr_y_test,arr_y_pred,figsize=None, bar_width=0.8,legend='on'):

	#computing the deciles
	deciles = decile_gen(arr_y_pred)

	#joining all the pieces together
	data = np.hstack((arr_y_test.reshape((len(arr_y_test),1)),
					  arr_y_pred.reshape((len(arr_y_pred),1)),
					  deciles.reshape((len(deciles),1))))

	data_df = pd.DataFrame(data)
	data_df.columns = ['actual','prob','decile']

	#computing the actual average across all of the observations
	total_avg_actuals = data_df.actual.mean()

	#aggregating on decile to find averages within decile buckets
	data_df = data_df.groupby('decile').agg({'actual':np.mean,'prob':np.mean}).reset_index()

	#plotting the data
	if figsize:
		plt.figure(figsize=figsize)
	#plt.hist(bin_means,bins=percentiles_linear_space)
	plt.bar(left=data_df.decile, 
	        height=data_df.prob, 
	        width = bar_width, 
	        align='center', 
	        color = 'blue', 
	        edgecolor = 'black', 
	        fill=True, 
	        alpha = 0.5)
	plt.scatter(data_df.decile,
	            data_df.actual, 
	            color = 'red', 
	            s=150)
	plt.axhline(y=total_avg_actuals, 
	            xmin=0, 
	            xmax=100, 
	            hold=None, 
	            color='black')
	plt.title('Average Decile Predicted Probability vs Actual Target Rate',
	          fontsize=20) 
	plt.xlabel('Probability Score Decile',
	           fontsize=15)
	plt.ylabel('Predicted Probability / Actual Target Rate',
	           fontsize=15)
	if legend=='on':
		plt.legend(['Baseline (Avg.) Response Rate','Actual Target Rate','Predicted Probability'],
		           fontsize=12)