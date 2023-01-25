import pandas, numpy
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Pool
from scipy.optimize import curve_fit as fit
import matplotlib
font = {'size'   : 8,
        "family":'serif'}

matplotlib.rc('font', **font)
plt.rcParams.update({
        "text.usetex": True})

fig_size_x, fig_size_y=8,8

def Sin(x, a,b,c,d):
    return a*numpy.sin(b*x-c)+d

def ThirdOrder(x, a,b,c,d):
    return a*x**3+b*x**2+c*x+d

def SecondOrder(x, a,b,c):
    return a*x**2+b*x+c

def MakeSteppedSpan(iterable, **kwargs):
    """
        take a minimum and maximum value from a list, 
        Use their difference for a width.
        From minimum to maximum value, create tuples that
        span width from minimum to maximum value.
        return tuples  
    """
    min_val = kwargs.pop("min_val", min(iterable))
    max_val = kwargs.pop("max_val",max(iterable))
    stepsize = kwargs.pop("stepsize",0)

    wholeIntervals= int(numpy.floor((max_val-min_val)/stepsize))-1
    spans = [[min_val+stepsize*i, min_val+stepsize*(i+1)-1] for i in range(wholeIntervals)]
    spans.append([min_val+stepsize*wholeIntervals,max_val])
    return spans

def MakeSplitSpan(iterable,**kwargs):
    """
        Iterable is the splitpoints,
        min and max values will be presumed from 
            iterable if not in kwargs.

        From minimum to maximum value, create tuples that
        span from minimum to maximum value.
        return tuples  

    """
    min_val = kwargs.pop("min_val", min(iterable))
    max_val = kwargs.pop("max_val",max(iterable))
    intervals = len(iterable)-1
    span=[]
    for i in range(intervals):
        if i == 0:
            span.append([min_val,iterable[i+1]-1])
            continue
        span.append([iterable[i],iterable[i+1]-1])

    return span

def GetNumber(name):
    # Grabs a number from the filename that follows the lab's
    #   file naming convention
    a = name.split('-')
    b = a[-1]
    c = b.split('.')
    print(c[0], name)
    return c[0]

def ParseFile(file, s, y="P124A (V)"):
    """
        Reads the file, corrects data if necessary (sensitivity scales everything such 
        that the final reported value is in Volts.)
    """
    # The correction factor is able to scale a specific
    #   stream of data by a constant. This is used ONLY
    #   when, during experiment, we don't enter the 
    #   lock-in amplifier's sensitivity, properly into the
    #   LABView program.
    correction = s[int(GetNumber(file))]
    
    with open(file, 'r') as f:
        df = pandas.read_csv(f, delimiter=',')
    if type(correction) ==float:
        df[y]=df[y].to_numpy()*correction

    return df

def SplitFile(*args,**kwargs):
    df= args[0]

    numWindows = kwargs.get("numWindows",3)
    B=kwargs.get('B',"B Field (T)")
    width=kwargs.get("width", int(len(df[B])/numWindows))
    degenerate=kwargs.get('degenerate',False)
    if degenerate:
        stepsize=kwargs.get('stepsize',1)

    # Define split-points for the DF based on the index
    #   Split points correspond to the nth datapoint.
    splits_for_df = numpy.arange(0,len(df[B])-1, width)
    
    # range_for_df is a nested-list [[start1, end1], ...]
    #   the difference between endX and start(X+1) is 1.
    #   No data is "sliced out" of analysis.
    
    if degenerate:
        range_for_df = MakeSteppedSpan(splits_for_df, len(df[B])-1, 50,degenerate=degenerate,width=width)
    else:
        range_for_df = MakeSplitSpan(splits_for_df,max_val=len(df[B])-1)

    return splits_for_df,range_for_df

def USelect(df,T,B,y,file,s):
    b=""
    while b !='y':
        # Untill the user is satisfied, allow them to select
        #       Data that they need.
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
        print("Window Width: ",max(xax)-min(xax))
        b = input("Ok? (y/n): ")
    Analyze(df,0,T,B,y,file,cut,other,s)

def anY1(df,idx,T,B,y,fn,cut,other,s):
    """
    anY1: Analyze Y1. or 1 stream of data.
    df: Pandas object
    idx: Slice number
    T: What's the temperature in the file
    B: What's the name of the B-Field in the file
    y: what's the y-data that we should look for
    fn: what's the file name being Analyzed
    cut: The data that's in the region we're going to FFT
    other: The data that's not cut.
    s: Sensitivity dictionary that we can use to scale
            our data by a constant incase we fail to record
            something prior to recording a run of data.

    """
    #fig, ax = plt.subplots(nrows=5,figsize=(8.5*7/11,11*7/11))
    fig, ax = plt.subplots(ncols=2,figsize=(11,4.25))
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
    #ax[1].set_xlabel('Tesla')

    """ax[2].scatter(cut[B], g, s=3, label="Double fourier transofrm")
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
    #ax[4].set_ylim(ym,yM)"""

    for i in ax:
        i.grid(True)
        i.legend(loc="best")
        i.set_xlabel("Tesla")
    
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

def Analyze():
    pass


def gff(df, function, **kwargs):
    #  https://github.com/Tristan-Anderson/Slifer_Lab_NMR_Toolsuite in NMR_Analyzer.py
    """
    Generalized Fitting Function


    """
    def get_function(f_name,xdata,var):
        if f_name == "Sin":
            yfit = Sin(xdata, var[0], var[1], var[2], var[3])

        elif f_name == "ThirdOrder":
            yfit = ThirdOrder(xdata, var[0], var[1], var[2], var[3])

        elif f_name == "SecondOrder":
            yfit = SecondOrder(
                xdata, var[0], var[1], var[2])
        
        return yfit

    # Get kw arguments
    # Sets which Y to evaluate
    y = kwargs.pop('y', "P124A (V)")

    # Sets which X to evaluage
    x = kwargs.pop('x', "B Field (T)")

    # Toggles saving the figure
    savefit = kwargs.pop('savefit', False)

    # If the above is toggled, the file needs a name
    filename = kwargs.pop('filename', 'UNNAMED_GRAPH')

    #automated?
    automated = kwargs.pop('automated', False)

    # selectregion?
    selectregion = kwargs.pop('selectregion', True)


    # Sets the window for data
    xmin = kwargs.pop('xmin', None)
    xmax = kwargs.pop('xmax', None)

    cut_data = pandas.DataFrame()
    if selectregion:
        fig, ax = plt.subplots(figsize=(fig_size_x, fig_size_y))
        ax.scatter(df[x],df[y],label=y,color='blue')
        ax.legend(loc='best')
        tuples = plt.ginput(2,timeout=False)        
        xlims = numpy.asarray([c[0] for c in tuples])
        cut_data = df[(df[x] > xlims[0]) & (df[x] < xlims[1])]

    elif xmin is not None and xmax is not None:
        cut_data = df[(df.x > xmin) & (df.x < xmax)]

    else:
        fit_data = df

    #print(df)

    fit_data = df.drop(cut_data.index.to_numpy())

    #print(fit_data)


    xdata = fit_data[x].values
    ydata = fit_data[y].values

        
    var, _ = fit(eval(function), xdata, ydata)
    # Fit that stuff
    fitname=y+" "+function+" fit"
    fitsub=fitname+" subtraction"

    df[fitname]=get_function(function, df[x], var)
    df[fitsub]=df[y]-df[y+" "+function+" fit"]

    fig, ax = plt.subplots(2,figsize=(fig_size_x, fig_size_y))

    ax[0].scatter(df[x],df[y],label='data',color='blue')
    ax[0].scatter(df[x],df[fitname],label='fit',color="red")

    ax[1].scatter(df[x],df[fitsub],label='fit subtraction',color="black")

    for i in ax:
        i.grid(True)
        i.legend(loc="best")

    print("#"*10)
    print("SELECT PEAKS OF OSCILLATIONS")
    print("#"*10)
    

    
    print(plt.ginput(2))

    
   
    return df


def StepWiseAnalysis(df,window, subwindows, filename,identifier, selectregion=False):
    #print(df)
    df = df.iloc[window[0]:window[1]]

    df = gff(df,"ThirdOrder",selectregion=selectregion)

    for sw in subwindows:
        s,f=sw[0],sw[1]
        cut= df.iloc[s:f]
        

    
def ABOscillation(filenames,s, **kwargs):
    B=kwargs.get('B',"B Field (T)")
    #Subwindow width in units of Tesla
    subwindowWidth=kwargs.get("subwindowWidth",5E-4)
    numWindows=kwargs.get("numWindows",3)
    for i in filenames:
        df = ParseFile(i,s)

        #print(df)
        # No overlap in splits.
        splitpoints, windows = SplitFile(df,numWindows)

        print("window indecies:",windows)

        #Find appropriate window subdivision (~5mT=5E-4)
        xax=df[B]
        dxax_NaN = xax.diff().to_numpy()
        dxax = dxax_NaN[~numpy.isnan(dxax_NaN)]
        b_stepsize = abs(dxax.mean())
        #print(b_stepsize, (max(xax)-min(xax))/(len(xax)))
        stepsPerSubWindow = int(numpy.ceil(subwindowWidth/b_stepsize))
        stepsPerGauss = numpy.ceil(1E-5/b_stepsize)

        # window = [start index df window:int, end index df window:int]
        for idx, w in enumerate(windows):
            
            splits_for_subwindows = numpy.arange(min(w),max(w), stepsPerSubWindow)
            #print(splits_for_subwindows)
            subwindows = MakeSteppedSpan(splits_for_subwindows, stepsize=int(stepsPerSubWindow/2))
            #StepWiseAnalysis(df, w, subwindows, i,idx)
            StepWiseAnalysis(df, w, subwindows, i,idx,selectregion=True)

            exit()

            


sensitivity = {1:[500*10**-6], 2:[500*10**-6], 3:[50*10**-3],
               4:[50*10**-6], 5:[10**-6], 6:[10**-6], 
               7:[500*10**-6], 8:[500*10**-6], 9:[1/500],
               10:[5*10**-6]}
sensitivity = {i:[1] for i in range(11)}
sensitivity[9] = [1/500]
sensitivity[10] = [1,1]       

#f = ["Aug26-Rings2-"+str(i)+".dat" for i in range(1,12)]
f = ["Aug26-Rings2-2.dat"]
#main(f,sensitivity)
ABOscillation(f,sensitivity)


"""
WHAT:  Analysis will use non-degenerate window-shifting, then subdivide each window into non-degenerate spans, of which 
            users will manually click on peaks to identify AB oscillations

HOW:
    1) Look at overall data trends.
    2) Elect number of windows (3) should do it -> but this is to be generalized.
    3) Fit subtract windows. -> keep track of fit paramaters per window.

d
    Window shifting:
            -> already done in MakeSpan.
    Subdivide window:
            -> Perhaps re-do it in
"""