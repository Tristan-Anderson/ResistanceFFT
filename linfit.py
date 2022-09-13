import numpy, pandas
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as fit

def line(x,m,b):
    return m*x+b

def main(f):
    with open(f, 'r') as df:
        df = pandas.read_csv(f)
    B = "B Field (T)"
    y = "P124A (V)"
    
    plt.scatter(df[B], df[y], label="Data")
    a = plt.ginput(2)
    
    xlim = [i[0] for i in a]
    
    fit_data = df[df[B]>min(xlim)]
    fit_data = fit_data[fit_data[B]<max(xlim)]
    print(fit_data)
    
    fig ,ax = plt.subplots(2)
    ax[0].scatter(df[B], df[y], label="All Data")
    ax[0].scatter(fit_data[B], fit_data[y], label="Selection")
    
    pcov,_ = fit(line, xdata=fit_data[B], ydata=fit_data[y])
    
    ax[1].scatter(fit_data[B], fit_data[y], label="Selection", color='red')
    f_data = line(fit_data[B].to_numpy(), pcov[0], pcov[1])
    ax[1].scatter(fit_data[B], f_data, label="Fit. m= "+str(round(pcov[0],10)), color="blue")
    for i in ax:
        i.legend(loc="best")
        i.grid(True)
    print(pcov[0], 'Is the slope of the graph')
    plt.show()

f = "Aug26-Rings2-4.dat"
main(f)
