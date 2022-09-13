import pandas, numpy
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Pool

"""
Test the DFFT. Make an x^2 thing with / sin(2pi x/(0.00058)) superimposed. Then test the thing.
"""

def makeSpan(iterable, max_val, stepsize, **kwargs):
    """
    takes an iterable and creates a list of lists that contain start/end
    to the unique values 
    """
    min_val = kwargs.pop("min_val", min(iterable))
    itr = iter(iterable)
    pairs = []
    for i, v in enumerate(itr):
        if v == max(iterable):
            pairs.append([v,max_val])
            if  pairs[-1][1]-pairs[-1][0]< stepsize-1:
                """
                if the very last tuple would return too small of a span,
                    make the penultimate span just a little bit larger
                """
                ef = pairs[-1][1]
                sr = pairs[-2][0]
                del pairs[-1]
                del pairs[-1]
                pairs.append([sr,ef])
            break
        else:
            pairs.append([iterable[i], iterable[i+1]-1])
    else:
        # if the above for loop executes fully (exhaustive iteration)
        # then the v==max(iterable) was never reached. If that happened
        # then the loop stopped running at the continue statement, meaning
        # the iterable passed was len < 1. Meaning only min-max is needed.
        pairs = [[min_val, max_val]]
    return pairs

def parsefile(file, stepsize=1500, T="Temp (K)", B="B Field (T)", y="P124A (V)", autocut=True):
    try:
        xlim = 5
        with open(file, 'r') as f:
            df = pandas.read_csv(f, delimiter=',')
        splits_for_df = numpy.arange(0,len(df[B])-1, stepsize)
        range_for_df = makeSpan(splits_for_df, len(df[B])-1, stepsize)
        for idx, i in enumerate(range_for_df):
            if autocut:
                analyze(i, df,idx,T,B,y, file)
            else:
                uselect(df,i, idx, T,B,y, file)
                print("Made it!")
    except Exception as e:
        print(e)
    return True

def uselect(df,i,idx,T,B,y,file):
    b=""
    while b !='y':
        plt.close('all')
        plt.plot(df[B], df[y], label="Data")
        plt.grid(True)
        plt.legend(loc='best')
    
        coords = plt.ginput(2)
        xax = [c[0] for c in coords]
        cut = df[(df[B]>max(xax)) & (df[B]<min(xax))]
        other = df[df[B]<max(xax)]
        other = other[other[B]>min(xax)]
        plt.close('all')
        fig,ax = plt.subplots(2)
        ax[0].scatter(other[B], other[y])
        ax[1].scatter(cut[B], cut[y])
        for i in ax:
            i.grid(True)
            i.legend(loc='best')
        plt.show()
        plt.close('all')
        b = input("Ok? (y/n): ")
    analyze(i,df,idx,T,B,y,file,cut,other)


def analyze(i,df,idx,T,B,y,fn,cut,other):
    fig, ax = plt.subplots(nrows=3)
    ax[0].scatter(other[B], other[y], label=y+" Cut "+str(idx),color="blue", s=3)
    ax[0].scatter(cut[B], cut[y], label="Data Subsection", color="red", s=3)
    ax[0].set_ylim(min(df[y]), max(df[y]))
    ax[0].set_xlim(min(df[B]), max(df[B]))
    
    ax[1].scatter(cut[B], cut[y], label="Cut", color="red", s=3)
    ax[1].set_ylim(min(cut[y]), max(cut[y]))
    ax[1].set_xlim(min(cut[B]), max(cut[B]))

    sr = 5000
    f = DFFT(cut[y])
    N = len(f)
    n = numpy.arange(N)
    T = N/sr 
    freq = n/T
    ax[2].set_xlabel('Freq (1/Tesla)')
    ax[2].stem(freq, abs(f),  label="FFT Frequencies", markerfmt=" ")
    ax[2].set_ylim(min(abs(f)), max(abs(f)))
    ax[2].set_xscale('log')

    for i in ax:
        i.grid(True)
        i.legend(loc="best")
    
    fig.suptitle("".join(list(fn)[:-4])+" Cut "+str(idx))
    plt.savefig("dump/"+"".join(list(fn)[:-4])+" Cut "+str(idx), dpi=200)
    plt.close('all')
    print("Completed", idx, fn)

def DFFT(x):
    N = len(x)
    n = numpy.arange(N)
    nt = n.reshape((N,1))
    e = numpy.exp(-2j*n*numpy.pi*nt/N)
    return numpy.dot(e,x)


def auto_analyze(i,df,idx,T,B,y, fn):
    cut = df.iloc[i[0]:i[1]]
    other = pandas.concat([df.iloc[:i[0]], df.iloc[i[1]:]])
    analyze(i,df,idx,T,B,y,fn)
    

def x(y):
    return y**2
def sin(y):
    return 1/32*numpy.sin(numpy.pi * 2*y/(0.00058))

def main(filenames):
    result_objects = []
    d = input("Y for Auto, N for manual")
    if d.upper()=="Y":
        for i in filenames:
            parsefile(i)
    else:
        for i in filenames:
            parsefile(i, autocut=False)
    exit()
    """cpus = int(3)
    with Pool(processes=cpus) as pool:#, initializer=start_process) 
        result_objects.append([pool.apply_async(parsefile, i) for i in filenames])
    pool.join()
    pool.close()"""
def main2(filenames):
    """
    s'pose now I want to take a file and select carefulyl where i want to do my analysis.
    def main2():
        open file
        get data
        plot data
        user selects data
        cut data
        analyze data
    """
    
f = ["Aug26-Rings2-"+str(i)+".dat" for i in range(8,9)]
main(f)
