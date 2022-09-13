import pandas, numpy
import matplotlib.pyplot as plt
"""
Test the DFFT. Make an x^2 thing with / sin(2pi x/(0.00058)) superimposed. Then test the thing.
"""

def main(filenames):
    df = ""
    T="Temp (K)"
    B = "B Field (T)"
    y1="P124A (V)"
    fig, ax = plt.subplots(nrows=3,ncols=2)
    xlim = 5
    for i,filename in enumerate(filenames):
        last = df
        with open(filename, 'r') as f:
            df = pandas.read_csv(f, delimiter=',')
        ax[i,0].scatter(df[B], df[y1], label=y1+" Run "+str(i), s=3)
        ax[i,0].legend(loc='best')
        ax[i,0].grid(True)
        #ax[i,0].set_xlim(-0.0182, 0.0182)
        ax[i,0].set_ylim(min(df[abs(df[B])<xlim][y1]), max(df[abs(df[B])<xlim][y1]))
        ax[i,0].set_xlim(min(df[abs(df[B])<xlim][B]), max(df[abs(df[B])<xlim][B]))
        # Fourier time 
        # Sample rate ~1000
        sr = 10000
        # Interval
        df = df[abs(df[B]) < 0.0182]
        print(df)
        f = DFFT(df[y1])
        N = len(f)
        n = numpy.arange(N)
        T = N/sr 
        freq = n/T
        
        ax[i,1].scatter(freq, abs(f), marker='s',label="Discrete FFT")
        ax[i,1].grid(True)
        ax[2,1].set_xlabel('Freq (1/Tesla)')
        ax[i,1].set_ylabel('DFT Amplitude |X(freq)|')

    fig.suptitle("Sweeps")
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


f = []
filename = "Aug26-Rings2-1.dat"
f.append(filename)
filename = "Aug26-Rings2-2.dat"
f.append(filename)
filename = "Aug26-Rings2-3.dat"
f.append(filename)
main(f)
"""
k = numpy.arange(0,1,0.005)
y = -x(k)+sin(k)

sr = 10000
# Interval
f = DFFT(y)
npfft=numpy.fft.fft(y)
N = len(f)
print(N)
n = numpy.arange(N)
T = N/sr 
freq = n/T
print(k)
print(freq)
fig, ax = plt.subplots(2)
ax[0].scatter(freq, abs(f), marker='s',label="Discrete FFT")
ax[0].grid(True)
ax[0].set_xlabel('Freq (Tesla)')
ax[0].set_ylabel('DFT Amplitude |X(freq)|')
ax[1].scatter(freq,abs(npfft), label="Signal")
for i in ax:
    i.legend(loc="best")
plt.show()
"""
"""
import lomb
x = numpy.arange(10)
y = numpy.sin(x)
fx,fy, nout, jmax, prob = lomb.fasper(x,y, 6., 6.)
plt.plot(fx,fy)
plt.show()
"""
