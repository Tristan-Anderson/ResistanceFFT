import pandas, numpy
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Pool
import matplotlib
font = {'size'   : 8,
        "family":'serif'}

matplotlib.rc('font', **font)
plt.rcParams.update({
        "text.usetex": True})

"""
Test the DFFT. Make an x^2 thing with / sin(2pi x/(0.00058)) superimposed. Then test the thing.
"""

def makeSpan(iterable, max_val, stepsize, **kwargs):
    """
    takes an iterable and creates a list of lists that contains
    start/end values that by default do not overlap.

    The degenerate flag allows for creation of regions with an
    approximately fixed width to shift through the data at a 
    given step-size

    """
    degenerate = kwargs.pop('degenerate', False)
    min_val = kwargs.pop("min_val", min(iterable))
    pairs = []
    if degenerate:
        width = kwargs.pop('width', 0)
        initial = min_val
        final = initial+width
        steps = int((max_val-width)/stepsize)
        a = [[min_val+stepsize*i, width+stepsize*i] for i in range(steps)]
        a[-1]=[a[-1][0], max_val]
        pairs=a
    else:
        itr = iter(iterable)
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

def getNumber(name):
    a = name.split('-')
    b = a[-1]
    c = b.split('.')
    print(c[0], name)
    return c[0]

def parsefile(file, s, width=1400, T="Temp (K)", B="B Field (T)", y="P124A (V)", autocut=True):
    correction = s[int(getNumber(file))]
    xlim = 5
    with open(file, 'r') as f:
        df = pandas.read_csv(f, delimiter=',')
    if type(correction) ==float:
        df[y]=df[y].to_numpy()*correction
    splits_for_df = numpy.arange(0,len(df[B])-1, width)
    range_for_df = makeSpan(splits_for_df, len(df[B])-1, 50,degenerate=True,width=1600)
    #range_for_df = makeSpan(splits_for_df, len(df[B])-1, width)
    p = multiprocessing.cpu_count()
    avp = int(p*.9)
    if autocut:
        with Pool(processes=avp) as pool:
            result_objs = [pool.apply_async(auto_analyze, args=(i,df,idx,T,B,y,file,s)) for idx, i in enumerate(range_for_df)]
            pool.close()
            pool.join()
        """
        for idx,i in enumerate(range_for_df):
            auto_analyze(i, df,idx,T,B,y, file)
        """
        results = [r.get() for r in result_objs]
    else:
        uselect(df, T,B,y, file,s)
    return True

def auto_analyze(i,df,idx,T,B,y, file,s):
    cut = df.iloc[i[0]:i[1]]
    other = pandas.concat([df.iloc[:i[0]], df.iloc[i[1]:]])
    analyze(df,idx,T,B,y,file, cut,other,s)
    return True

def uselect(df,T,B,y,file,s):
    b=""
    while b !='y':
        plt.close('all')
        plt.plot(df[B], df[y], label="Data")
        plt.grid(True)
        plt.legend(loc='best')
    
        coords = plt.ginput(2)
        xax = [c[0] for c in coords]
        cut = df[(df[B]<max(xax)) & (df[B]>min(xax))]
        other = df[df[B]>max(xax)]
        other = pandas.concat([other, df[df[B]<min(xax)]])
        print(other)
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
    analyze(df,0,T,B,y,file,cut,other,s)

def anY1(df,idx,T,B,y,fn,cut,other,s):
    fig, ax = plt.subplots(nrows=5,figsize=(8.5*7/11,11*7/11))
    ax[0].scatter(other[B], other[y], label=y+" Cut "+str(idx),color="blue", s=3)
    ax[0].scatter(cut[B], cut[y], label="Data Subsection", color="red", s=3)
    ax[0].set_ylim(min(df[y]), max(df[y]))
    ax[0].set_xlim(min(df[B]), max(df[B]))
    
    ax[1].scatter(cut[B], cut[y], label="Cut", color="red", s=3)
    ax[1].set_ylim(min(cut[y]), max(cut[y]))
    ax[1].set_xlim(min(cut[B]), max(cut[B]))

    sr = 5000
    f = DFFT(cut[y])

    g = numpy.real(iDFFT(f))
    ax[2].scatter(cut[B], g, s=3, label="Double fourier transofrm")
    ax[3].scatter(cut[B], cut[y].to_numpy()-g, s=3, label="Diff Actual v. Fourier Tform")


    N = len(f)
    n = numpy.arange(N)
    T = N/sr 
    freq = n/T
    ax[4].set_xlabel('Freq (1/Tesla)')
    ax[4].stem(freq, abs(f),  label="FFT Frequencies", markerfmt=" ")

    cc = pandas.DataFrame({B:freq, y:abs(f)})
    xm, xM = 1000, 1500
    ax[4].plot([xm,xM],[0, max(cc[y])], color='red', label="Where our signal should be")
    ax[4].plot([xM,xm],[0, max(cc[y])], color='red')
    cc = cc[cc[B]>xm]
    cc = cc[cc[B]<xM]
    ym, yM = min(cc[y]), max(cc[y])

    #ym, yM = min(abs(f[2:-2])), max(abs(f[2:-2]))
    ax[4].set_xscale('log')
    ax[4].set_yscale('log')
    #ax[4].set_xlim(xm,xM)
    #ax[4].set_ylim(ym,yM)

    for i in ax:
        i.grid(True)
        i.legend(loc="best")
    
    fig.suptitle("".join(list(fn)[:-4])+" Cut "+str(idx))
    plt.tight_layout()
    plt.savefig("dump/"+"".join(list(fn)[:-4])+"-Cut-"+"%5.5i"% (idx), dpi=150)
    plt.close('all')
    print("Completed", idx, fn)

def anYN(df,idx,T,B,yn,fn,cut,other,s, others_ys=["SR830 1 X (V)"]):
    others_ys.append(yn)
    unique = []
    for i in others_ys:
        if i not in unique:
            unique.append(i)
    fig, ax = plt.subplots(nrows=5,ncols=len(unique),figsize=(8.5*7/11*len(unique),11*7/11))
    for column, y in enumerate(unique):
        ax[0,column].scatter(other[B], other[y], label=y+" Cut "+str(idx),color="blue", s=3)
        ax[0,column].scatter(cut[B], cut[y], label="Data Subsection", color="red", s=3)
        ax[0,column].set_ylim(min(df[y]), max(df[y]))
        ax[0,column].set_xlim(min(df[B]), max(df[B]))
        
        ax[1,column].scatter(cut[B], cut[y], label="Cut", color="red", s=3)
        ax[1,column].set_ylim(min(cut[y]), max(cut[y]))
        ax[1,column].set_xlim(min(cut[B]), max(cut[B]))
    
        sr = 5000
        f = DFFT(cut[y])
    
        g = numpy.real(iDFFT(f))
        ax[2,column].scatter(cut[B], g, s=3, label="Double fourier transform")
        ax[3,column].scatter(cut[B], cut[y].to_numpy()-g, s=3, label="Diff Actual v. Fourier Tform")
    
    
        N = len(f)
        n = numpy.arange(N)
        T = N/sr 
        freq = n/T
        ax[4,column].set_xlabel('Freq (1/Tesla)')
        ax[4,column].stem(freq, abs(f),  label="FFT Frequencies", markerfmt=" ")
    
        cc = pandas.DataFrame({B:freq, y:abs(f)})
        xm, xM = 1000, 1500
        ax[4,column].plot([xm,xM],[0, max(cc[y])], color='red', label="Where our signal should be")
        ax[4,column].plot([xM,xm],[0, max(cc[y])], color='red')
        cc = cc[cc[B]>xm]
        cc = cc[cc[B]<xM]
        ym, yM = min(cc[y]), max(cc[y])
    
        ax[4,column].set_xscale('log')
        ax[4,column].set_yscale('log')

    for i in ax:
        for j in i:
            j.grid(True)
            j.legend(loc="best")
    dpi = int(1600/(8.5*7/11*len(unique)))
    
    fig.suptitle("".join(list(fn)[:-4])+" Cut "+str(idx))
    plt.tight_layout()
    plt.savefig("dump/"+"".join(list(fn)[:-4])+"-Cut-"+"%5.5i"% (idx), dpi=dpi)
    plt.close('all')


def analyze(df,idx,T,B,y,fn,cut,other,s):
    num = int(getNumber(fn))
    if len(s[num]) == 1:
        anY1(df,idx,T,B,y,fn,cut,other,s)
    else:
        anYN(df,idx,T,B,y,fn,cut,other,s)

def DFFT(x):
    N = len(x)
    n = numpy.arange(N)
    nt = n.reshape((N,1))
    e = numpy.exp(-2j*n*numpy.pi*nt/N)
    return numpy.dot(e,x)

def iDFFT(x):
    N = len(x)
    n = numpy.arange(N)
    nt = n.reshape((N,1))
    e = 1/N*numpy.exp(2j*n*nt*numpy.pi/N)
    return numpy.dot(e,x)
    

def x(y):
    return y**2

def sin(y):
    return 1/32*numpy.sin(numpy.pi * 2*y/(0.00058))

def main(filenames, s):
    result_objects = []
    d = input("Y for Auto, N for manual: ")
    if d.upper()=="Y":
        for i in filenames:
            parsefile(i, s)
    else:
        for i in filenames:
            parsefile(i, s,autocut=False)
    exit()
    
sensitivity = {1:[500*10**-6], 2:[500*10**-6], 3:[50*10**-3],
               4:[50*10**-6], 5:[10**-6], 6:[10**-6], 
               7:[500*10**-6], 8:[500*10**-6], 9:[1/500],
               10:[5*10**-6]}
sensitivity = {i:[1] for i in range(11)}
sensitivity[9] = [1/500]
sensitivity[10] = [1,1]

f = ["Aug26-Rings2-"+str(i)+".dat" for i in range(1,12)]
main(f,sensitivity)
