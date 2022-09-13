import pandas, numpy
import matplotlib.pyplot as plt
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

def parsefile(file, stepsize=170, T="Temp (K)", B="B Field (T)", y="P124A (V)"):
    xlim = 5
    with open(file, 'r') as f:
        df = pandas.read_csv(f, delimiter=',')
    splits_for_df = numpy.arange(0,len(df[B])-1, stepsize)
    range_for_df = makeSpan(splits_for_df, len(df[B])-1, stepsize)
    for idx, i in enumerate(range_for_df): 
        analyze(i, df,idx,T,B,y)

def analyze(i,df,idx,T,B,y):
    cut = df.iloc[i[0]:i[1]]
    other = pandas.concat([df.iloc[:i[0]], df.iloc[i[1]:]])
    print(other)
    print([k for k in other])
    print(cut)
    print([k for k in cut])
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
    ax[2].scatter(freq, abs(f), label="FFT Frequencies", color="m", s=3, marker='o')
    
    fig.suptitle("Cut "+str(idx))
    plt.show()

def DFFT(x):
    N = len(x)
    n = numpy.arange(N)
    nt = n.reshape((N,1))
    e = numpy.exp(-2j*n*numpy.pi*nt/N)
    return numpy.dot(e,x)


def x(y):
    return y**2
def sin(y):
    return 1/32*numpy.sin(numpy.pi * 2*y/(0.00058))

def main(filenames):
    df = ""
    for i in f:
        parsefile(i)

f = []
filename = "Aug26-Rings2-1.dat"
f.append(filename)
filename = "Aug26-Rings2-2.dat"
f.append(filename)
filename = "Aug26-Rings2-3.dat"
f.append(filename)
main(f)
