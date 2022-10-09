# Libraries
# PLEASE USE PYTHON3 WHEN STARTING PROGRAM
import numpy as np 
import pandas as pd
import json
import scipy.optimize as sco
import scipy.interpolate as sci
import matplotlib.pyplot as plt
import plotly.graph_objects as objg
from nltk.sentiment.vader import SentimentIntensityAnalyzer   
import nltk
from sklearn.preprocessing import minmax_scale
import seaborn as sns

# DEFINES
senAnalyser = SentimentIntensityAnalyzer()

# ////////////////////////////////////////////////////
# ///////////           FUNCTIONS       //////////////
# ////////////////////////////////////////////////////

# Station 1!! :)
def assetSetting():
    second_sheet = pd.ExcelFile('ASX200top10.xlsx')
    stocks = pd.read_excel(second_sheet, 'Bloomberg raw', header = [0,1], index_col = 0)

    # Make sure to keep only stock price and labels
    fixed_stocklist = set(stocks.columns.get_level_values(0))
    temp = stocks.iloc[:, stocks.columns.get_level_values(1) == 'PX_LAST']
    temp.columns = temp.columns.droplevel(1)

    asx_stocks = temp["AS51 Index"]
    temp = temp.drop("AS51 Index", axis = 1)

    return asx_stocks, temp



# SENTIMENT ANALYSER - USES THE NEWS DUMP DATA TO OBTAIN EQUITY
def newsFiltering():
    # This requires the filtering of the news_dump.json
    # This encompasses the stations of ETL and FE (1 and 2)
    news = pd.read_json("news_dump.json")

    # Filter and clean the data
    news["Date/Time"] = pd.to_datetime(news["Date/Time"])

    # Proceed to Station 3: MD (filter out and keep scores appropriately)
    # Produces a lists of dictionary of scores in relation to sentiment analyser and clean into a proper format
    s = news["Headline"].apply(senAnalyser.polarity_scores).tolist()
    news = news.join(pd.DataFrame(s), rsuffix="_right")

    # Calculate the mean sentiment, filter and tranpose
    mean = news.groupby(["Equity"]).mean()[["compound"]].T
    
    lastest_month_score = mean.tail(20).mean()
    lastest_month_score.fillna(0,inplace=True)
    lastest_month_score.plot(kind='bar')
    
    plt.title('Sentiment Score')
    plt.grid()
    plt.show()
    plt.savefig("sen_ana2.png")

    average_score = lastest_month_score.mean()
    lastest_month_score = (lastest_month_score - average_score)
    lastest_month_score.plot(kind='bar')
    plt.title('Adjustment on Portfolio Weights -- Sentiment Score')
    plt.grid()
    plt.show()
    plt.savefig("adjused_sen.png")

    return lastest_month_score, news


# Calculates the returns, volatility, weights, sharpe.
def dataExtraction(stock, test_limit, news):
    # Initialise all relevant lists of returns, volumes, weights and sharpe ratio
    rets = np.log(stock / stock.shift(1))
    noa = len(stock.columns)
    prets = []
    pvol = []
    pweights = []
    psharpe = []


    # Find the adjusted returns first!
    returns = np.array([])
    return_fixed = 0
    if news == "yes":
        adjusted, rand = newsFiltering()

        for idx in range(len(adjusted)):
            returns = np.append(returns, adjusted[idx] * rets.mean()[idx] + rets.mean()[idx])

    else:
        returns = rets.mean()
    

    # Then calculate the amount for each and append to respective list
    for i in range(test_limit):
        weight = np.random.random(noa)
        weight /= np.sum(weight)
        pweights.append(weight)

        finalPortRet = np.sum(returns * weight) * 252
        prets.append(finalPortRet)

        calc_vol = np.sqrt(np.dot(weight.T, np.dot(rets.cov() * 252, weight)))
        pvol.append(calc_vol)

        calc_sharpe = (finalPortRet - 0.002591) / calc_vol
        psharpe.append(calc_sharpe)

    calc = np.array([finalPortRet, calc_vol, finalPortRet / calc_vol])

    return prets, pvol, pweights, psharpe, calc, returns



def minArr(w, rets):
    return overallCalc(w, rets)[1]

def maxSharpe(w, rets):
    return -1 * overallCalc(w, rets)[2]

def minVar(w, rets):
    return overallCalc(w, rets)[1] ** 2



# ////////////////////////////////////////////////////
# ///////////           MAIN            //////////////
# ////////////////////////////////////////////////////

# Introduction to user
print("Welcome to PPC! This program aims to provide an automated advanced portfolio management product. Before we start generating, a few questions are required to personalise the portfolio.")
risk_tol = input("Please enter your risk tolerance. The risk tolerance must be between 0-10: ")
risk_tol = float(risk_tol)
if not risk_tol >= 0 and not risk_tol <= 10:
    print("Incorrect input. The program will now exit.")
    exit(0)

comp_ran = input("Please enter the number of scearios to be generated. (Note, the higher the number, the more times it takes to process.): ")
news_option = input("Would you like to include news sentiment analysis into the portfolio? Please answer either with a 'yes' or 'no': ")

if not news_option == "yes" and not news_option == "no":
    print("Incorrect input. The program will now exit.")
    exit(0)

# Was meant to be used by never did :S
capital = input("Please enter your level of capital: ")


asx, stock_list = assetSetting()
noa = len(stock_list.columns)

# Return data
prets, pvol, pweights, psharpe, overall, adj_ret = dataExtraction(stock_list, int(comp_ran), news_option)

# Graph creation
df = pd.DataFrame(pweights, columns = stock_list.columns)
df["returns"] = np.array(prets)
df["volatility"] = np.array(pvol)
df["sharpe_ratio"] = np.array(psharpe)


# Redefine proper values
pvol = np.array(pvol)
psharpe = np.array(psharpe)
prets = np.array(prets)
recalc_weights = np.random.random(noa)
recalc_weights /= np.sum(recalc_weights)
rets = adj_ret
rets3 = np.log(stock_list / stock_list.shift(1))

# Histogram produced
rets3.hist(bins = 50, figsize = (15,12))
plt.show()
plt.savefig("histo.png")

# Calculate proper weightings again for future use
def overallCalc(weights, rets1):
    weights = np.array(weights)
    pret = np.sum(rets1 * weights) * 252
    pvol = np.sqrt(np.dot(weights.T, np.dot(rets3.cov() * 252, weights)))
    return_arr = np.array([pret, pvol, pret / pvol])
    return return_arr


# Find max and min points of sharpe index and volatlity respectively
sharpe_max = df["sharpe_ratio"].idxmax()
vol_min = df["volatility"].idxmin()

# Now fix weighting and find max/min risk
volatility_adjustments = df["volatility"].max() - df["volatility"].min()

# Convert the file to find consumer sentimental index
senGet = pd.read_excel("Economic_Indicators.xlsx",header = 0)
senGet = senGet.iloc[22].T


# Station 3 :)
# Plot the figure so far with the amount of simulations given
plt.figure(figsize=(18, 10))
plt.scatter(pvol, prets, c = prets / pvol, marker = 'o')
plt.grid(True)
plt.ylabel("exp rets")
plt.xlabel("exp vol")
plt.colorbar(label = "Sharpe Ratio")
plt.show()
plt.savefig("temp.png")

# Heatmap
fig, ax = plt.subplots(figsize=(18,10))
heatmap = sns.heatmap(rets3.cov() * 252, cmap = "coolwarm", annot = True)
figure = heatmap.get_figure()
figure.savefig("heatmap.png")

# Develop a new graph portfolio, minimising std for each return and now graph
binds = tuple((0,1) for x in recalc_weights)
rets2 = np.linspace(0.0, 0.25, 50)
vol2 = []
for i in rets2:
    cons = ({'type': 'eq', 'fun': lambda x:  overallCalc(x, rets)[0] - i},
            {'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})
    res = sco.minimize(minArr, noa * [1. / noa,], rets, method='SLSQP', 
                        bounds = binds, constraints = cons)
    vol2.append(res["fun"])

vol2 = np.array(vol2)


# Portfolio with max sharpe ratio
cons = ({'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})
bnds = tuple((0, 1) for x in range(noa))

opts = sco.minimize(maxSharpe, noa * [1. / noa,], rets, method='SLSQP',
                    bounds=bnds, constraints=cons)


# Portfolio with min var
optv = sco.minimize(minVar, noa * [1. / noa,], rets, method='SLSQP',
                    bounds=bnds, constraints=cons)

            
# Plot the figure of random portfolio composition in comparison to other, along with the EF and Sharpe Ratio
plt.figure(figsize = (18,10))
plt.scatter(pvol, prets, c = prets / pvol, marker = "o")
plt.scatter(vol2, rets2, c = rets2 / vol2, marker = "x")
t1 = overallCalc(opts["x"], rets)[1]
t2 = overallCalc(opts["x"], rets)[0]
w1 = overallCalc(optv["x"], rets)[1]
w2 = overallCalc(optv["x"], rets)[0]
plt.plot(t1, t2, "r*", markersize = 15.0)
plt.plot(w1, w2, "y*", markersize = 15.0)
plt.grid(True)
plt.ylabel("exp rets")
plt.xlabel("exp vol")
plt.colorbar(label = "Sharpe Ratio")
plt.show()
plt.savefig("temp2.png")


# Calculate the EF and CML
tck = sci.splrep(sorted(vol2[np.argmin(vol2):]), rets2[np.argmin(vol2):])

def lineCalc(p1):
    l1 = 0.032 - p1[0]
    l2 = 0.032 + p1[1] * p1[2] - sci.splev(p1[2], tck, der = 0)
    l3 = p1[1] - sci.splev(p1[2], tck, der = 1)

    return l1, l2, l3
line_points = sco.fsolve(lineCalc, [0.01, 0.5, 0.15])

# Calculate optimal tangent
cons = ({'type': 'eq', 'fun': lambda x:  overallCalc(x, rets)[0] - sci.splev(line_points[2], tck, der = 0)},
         {'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})
bnds = tuple((0, 1) for x in recalc_weights)
res = sco.minimize(minArr, noa * [1. / noa,], rets, method='SLSQP', 
                    bounds = binds, constraints = cons)


# Now plot everything together, with the capital market line and EF
plt.figure(figsize = (18,10))
plt.scatter(pvol, prets, c = prets / pvol, marker = "o")

plt.plot(vol2[np.argmin(vol2):], rets2[np.argmin(vol2):], "g", lw = 4.0)

t1 = np.linspace(0.0, 0.3)
plt.plot(t1, line_points[0] + line_points[1] * t1, lw = 1.5)

plt.plot(line_points[2], sci.splev(line_points[2], tck, der = 0), 'r*', markersize=15.0)
plt.grid(True)
plt.ylabel("exp rets")
plt.xlabel("exp vol")
plt.colorbar(label = "Sharpe Ratio")
plt.axhline(0, color='k', ls='--', lw=2.0)
plt.axvline(0, color='k', ls='--', lw=2.0)
plt.show()
plt.savefig("temp3.png")

# Station 4 :)
print("The maximisation of the Sharpe Ratio returns:\n" + str(opts["x"].round(3)) + "\n" + str(overallCalc(opts["x"], rets).round(3)))
print("The minimisation of portfolio Variance returns:\n" + str(optv["x"].round(3)) + "\n" + str(overallCalc(optv["x"], rets).round(3)))
print("The Efficient Frontier are represented as: " + str(line_points[0]) + ", " + str(line_points[1]))
print("The Capital Market Line can be represented by: " + str(line_points[2]))
print(str(np.round(lineCalc(line_points), 6)))
print("The Optimal Tangent portfolio follows: " + str(res["x"].round(3)))



# Now to adjust portfolio weights and print for client accordingly
client_deets = pd.read_excel("Client_Details.xlsx", sheet_name = "Data", index_col = 0)
client_deets["risk_profile"] = 11 - client_deets["risk_profile"]
for i in client_deets["risk_profile"]:
    y = (line_points[2] - 0.032) / (i * (float(sci.splev(line_points[2], tck, der = 0)) ** 2))
    
    cons = ({'type': 'eq', 'fun': lambda x:  overallCalc(x, rets)[0] - sci.splev(line_points[2], tck, der = 0)},
          {'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})
    bnds = tuple((0, 1) for x in recalc_weights)
    res = sco.minimize(minArr, noa * [1. / noa,], rets, method='SLSQP', 
                    bounds = binds, constraints = cons)
    
    p1 = res["x"] * max(y, 0)
    fixed = p1.round(3)

    print("This client is recommended to invest %.3lf in the risk-free asset and the remaining in the following portfolios:" % max(1 - y, 0))
    print((res['x'] * max(y,0)).round(3))


