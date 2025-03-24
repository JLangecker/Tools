# This script reads asset returns form a CSV file and
# performs mean variance optimization on them to create an optimal portfolio
# for given target returns.

# Import the libraries
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from pypfopt import efficient_frontier
import statistics
import math

# variables
returns_dict = {} # dictionary of lists of returns for each imported asset
mean_returns = [] # list of mean returns for each imported asset
stdev = [] # list of standard deviations for each imported asset
target_returns = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1] #change this to personal preference
cov_matrix = [] # initialized further down
volatility = [] # list of volatilities for each generated portfolio
sharpe = [] # list of sharpe ratios for each generated portfolio
weights_dict = {} # dictionary of optimal portfolio  (in a list) for each target return
portfolio_dict = {} # dictionary of optimal portfolio weights for each target return and the corresponding volatility

# define a covariance, mean and stdev functions that work with lists of unequal size (aka lists that have nan values anywhere)
def covariance(list_1, list_2):
    print("covariance function called")
    print(len(list_1))
    print(len(list_2))

    sum = 0
    for i in range(len(list_1)):
        if not math.isnan(list_1[i]) and not math.isnan(list_2[i]):
            sum += (list_1[i] - arithmetic_mean(list_1)) * (list_2[i] - arithmetic_mean(list_2))
        else:
            print("List1: ", list_1[i])
            print("List2: ", list_2[i])
            continue

    print("Sum: ", sum)

    # get length of shorter list exluding NaN entries
    list_1 = [list_1[i] for i in range(len(list_1)) if not math.isnan(list_1[i])]
    list_2 = [list_2[i] for i in range(len(list_2)) if not math.isnan(list_2[i])]

    shortest_list_length = len(list_1)
    if len(list_1) < len(list_2):
        shortest_list_length = len(list_1)
    elif len(list_2) < len(list_1):
        shortest_list_length = len(list_2)

    cov = sum/shortest_list_length

    return cov

def arithmetic_mean(list):

    sum = 0
    for i in range(len(list)):
        if not math.isnan(list[i]):
            sum += list[i]
        else:
            continue

    # get length of list exluding NaN entries
    list = [list[i] for i in range(len(list)) if not math.isnan(list[i])]
    actual_list_length = len(list)

    sum = sum/actual_list_length

    return sum

def standard_deviation(list):
    
    sum = 0
    for i in range(len(list)):
        if not math.isnan(list[i]):
            sum += (list[i] - arithmetic_mean(list))**2
        else:
            continue
    
    # get length of list exluding NaN entries
    list = [list[i] for i in range(len(list)) if not math.isnan(list[i])]
    actual_list_length = len(list)

    stdev = math.sqrt(sum/actual_list_length)
    return stdev

# import annual returns for each asset from CSV, and store them in a dictionary of lists
# Each column in the represents one asset with the returns for each year in the rows
# IMPORTANT: The returns need to be in this format: 0.05 means 5% per annum (if you just write 5.0, then the algorithm will think the asset produces 500% return and it will hugely overvalue that asset)
relative_filepath = "data/Asset_Returns_Data.csv"
returns_df = pd.read_csv(relative_filepath)

# get list of original asset names
original_asset_names = returns_df.columns.to_list()

# transform the names of the assets into Asset1 / 2 / 3 (because some demon has possessed me to use a dictionaries here and I don't want to rewrite the entire program)
current_columns = returns_df.columns
new_column_names = {current_columns[i]: f'Asset{i+1}' for i in range(len(current_columns))}
for i in range(returns_df.shape[1]):
    returns_df.rename(columns=new_column_names, inplace=True)
returns_dict = returns_df.to_dict(orient='list')


# calculate the mean and standard deviation for each asset
for i in range(len(returns_dict)):
    mean_returns.append(arithmetic_mean(returns_dict[f'Asset{i+1}']))
    stdev.append(standard_deviation(returns_dict[f'Asset{i+1}']))
    

cov_matrix = [[0 for i in range(len(returns_dict))] for j in range(len(returns_dict))]
# compute the covariance matrix    
for i in range(len(returns_dict)):
    print("Row: ", i)
    for j in range(len(returns_dict)):
        print("Column: ", j)
        cov_matrix[i][j] = covariance(returns_dict[f'Asset{i+1}'], returns_dict[f'Asset{j+1}'])
print(cov_matrix)

# turn cov_matrix into a dataframe (required by pypfopt), and create a list of the keys of the returns_dict
cov_matrix = pd.DataFrame(cov_matrix)
keys_list = list(returns_dict.keys())

# print mean returns and stdev of the imported assets
print("Mean Returns: ", end="\n")
for i in range(len(returns_dict)):
    print(keys_list[i], ":", mean_returns[i], end="\n")
print("\n")
print("Standard Deviations: ", end="\n")
for i in range(len(returns_dict)):
    print(keys_list[i], ":", stdev[i], end="\n")
print("\n")

# implement Mean Variance Optimization with ppo to calculate the optimal portfolio weights
for target_return in target_returns:
    try:
        ef = efficient_frontier.EfficientFrontier(mean_returns, cov_matrix)
        ef.efficient_return(target_return)
        weights = ef.clean_weights() # get the optimal portfolio weights for the target return
        _, volatility_i, sharpe_i = ef.portfolio_performance() # get the volatility and sharpe ratio of the generated portfolio
        volatility.append(volatility_i)
        sharpe.append(sharpe_i)
    except:
        print("Error: The target return " + str(target_return) + " is not possible with the given data.")
        break

    # print the optimal portfolio weights for each target return
    # also print the volatility and sharpe ratio of the generated portfolios
    print("Asset allocation for target return: ", target_return)
    for i in range(len(returns_dict)):
        print(keys_list[i], ":", weights[i])

    print("Portfolio - volatility: ", volatility[target_returns.index(target_return)])
    print("Sharpe Ratio: ", sharpe[target_returns.index(target_return)], "\n")

    # save the weights in a dictionary
    weights_dict["Return: " + str(target_return)] = weights


# generate an exponential fit for the volatilities as a function of the target returns
# and plot the results

# define the exponential function (Volatility as function of target return)
def f(x,a,b):
    return a* np.exp(float(b)*x)

# fit the exponential function to the data and plot the results
param_opt, cov_opt = sp.optimize.curve_fit(f, target_returns[0:len(volatility)], volatility)
plt.plot(target_returns[0:len(volatility)], volatility,'o', label='data') # there must be the same number of x and y values. Thus, we use the first len(volatility) values of target_returns since not all volatilites can be calculated as not all target returns are possible

print("Optimal parameters: ", param_opt)

# plot the fit for a more continuous range of x values
x_fit = np.linspace(min(target_returns),max(target_returns),100) # generate a range of x values to plot the fit
plt.plot(x_fit, f(x_fit, param_opt[0], param_opt[1]), label='fit')


plt.title('Volatility as a function of Target Returns')
plt.legend()
plt.show()

# export portfolio structures for each target return and their corresponding volatilities
# into a CSV file

# add the volatlity for each portfolio to the dictionary of optimal portfolio weights
i = 0
for key in weights_dict:
    portfolio_dict[str(key)] = list(weights_dict[key].values())
    portfolio_dict[str(key)].append(volatility[i])
    i += 1

# convert the dictionary to a dataframe and export it to a CSV file  
relative_filepath = "data/Example_Portfolios.csv"  
portfolio_df = pd.DataFrame(portfolio_dict)

# add new columns to dataframe with individual asset returns and volatilites, and original asset names
mean_returns.append(0)
portfolio_df['return'] = mean_returns
stdev.append(0)
portfolio_df['volatility / stdev'] = stdev
original_asset_names.append(0)
portfolio_df['original_asset_names'] = original_asset_names



portfolio_df.to_csv(relative_filepath, index=False)
# the CSV contains the target return and the weight of each asset in the portfolio for that target
# return as well as the volatility, and original asset names
# One column is structured as follows: (Target return, asset1, asset2, asset3, asset4, portfolio volatility)
# the last three columns contain the mean returns per asset, the stdev per asset and asset name

#also export the covariance matrix
relative_filepath = "data/Covariance_Matrix.csv"
cov_matrix_df = pd.DataFrame(cov_matrix)
cov_matrix_df.to_csv(relative_filepath, index = False)
# the first row is not part of the matrix and is added automatically by pandas for legibiltiy


