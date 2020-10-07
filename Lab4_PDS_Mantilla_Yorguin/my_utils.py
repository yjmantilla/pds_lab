import matplotlib.pyplot as plt
import numpy as np

def plot_signal(x,y=None,xlabel=None,ylabel=None,title=None,format=None,size=(15,5),show=True,ret=False,subplots=(1,1),stem=False):
    fig, axs = plt.subplots(*subplots)
    if isinstance(axs,np.ndarray):
        ax = axs[0]
    else:
        ax = axs
    ax.set_aspect('auto')

    if stem:
        foo = ax.stem
    else:
        foo = ax.plot
    if y is not None:
        if format is not None:
            foo(x,y,format)
        else:
            foo(x,y)
    else:
        if format is not None:
            foo(x,format)
        else:
            foo(x)
    if xlabel is not None:
        ax.set_xlabel(xlabel,fontsize=18)
    if ylabel is not None:
        ax.set_ylabel(ylabel,fontsize=18)
    if title is not None:
        ax.set_title(title,fontsize=24)
    if size is not None:
        fig.set_size_inches(size)
    plt.grid()
    plt.tight_layout()
    if show:
        plt.show()
    if ret:
        if subplots == (1,1):
            return fig,ax
        else:
            return fig,axs

def s2f(seconds,samplerate=48000):
    if isinstance(seconds,np.ndarray):
        frames = seconds*samplerate
        return frames.astype(int)
    else:
        return int(seconds*samplerate)

def f2s(frames,samplerate=48000):
    if isinstance(frames,np.ndarray):
        seconds = np.divide(frames,samplerate)
        return seconds
    else:
        return frames/samplerate


def find_nearest(array, values):
    array = np.asarray(array)
    idx = [(np.abs(array - value)).argmin() for value in values]
    return idx

def get_harmonic_signal(freqs,weights=None,phases=None,s_rate=1000,start=0,stop=10,withtime=False,time=None):
    """
    Function that returns a harmonic signal given its components.
    Parameters:
        freqs: list
                the frequencies [Hz] of the components
        weights: list
                the weights  of the components
        phases: list
                the phases of the components
        s_rate: scalar
                the sampling rate
        start: scalar
                start time
        stop: scalar
                stop time
        time : np_array
    
    Returns:
        harmonic_signal: numpy.ndarray
                the harmonic signal generated
    Testing:
        s_rate = 1000
        length = s_rate
        freqs_sin = np.array([10,10])
        weights_sin = np.array([1,1])
        phases_sin = np.array([0,180])
        start = 0
        stop = 10
        sin = get_harmonic_signal(freqs_sin,weights_sin,phases_sin,s_rate,start,stop)
        You should get an almost null vector (this is a really bad example)
    """
    freqs = as_list(freqs)
    if weights is None:
        weights = [1]*len(freqs)
    if phases is None:
        phases = [0]*len(freqs)
    weights = as_list(weights)
    phases = as_list(phases)
    if time is None:
        time = np.arange(start, stop, 1/s_rate)
    time = np.squeeze(time)
    harmonic_signal = np.zeros(time.shape[0])
    rad_phases = [phase * np.pi/180 for phase in phases]
    rad_freqs = [2 * np.pi * freq for freq in freqs]
    sinusoid = np.zeros((len(freqs),time.shape[0]))
    for i in range(0,len(freqs)):
        sinusoid[i,:] = weights[i] * np.sin(rad_freqs[i] * time + rad_phases[i])

    harmonic_signal = np.sum(sinusoid,axis=0)
    if withtime == True:
        return time,harmonic_signal
    else:
        return harmonic_signal


def as_list(x):
    if type(x) is list:
        return x
    elif type(x) is np.ndarray:
        return x.tolist()
    else:
        return [x]


def quantize_by_delta(x,delta,mode='round'):
    if mode == 'round':
        def foo(x):
            return delta * np.round(x/delta)
    if mode == 'mid-tread':
        def foo(x):
            return delta * np.floor(0.5 + x/delta)
    if mode == 'mid-riser':
        def foo(x):
            return delta * (0.5 + np.floor(x/delta))
    fun = np.vectorize(foo)
    return fun(np.asarray(x))

def my_quantize(vals, to_values,mode='round'):
    """Quantize a value with regards to a set of allowed values.
    
    Examples:
        quantize(49.513, [0, 45, 90]) -> 45
        quantize(43, [0, 10, 20, 30]) -> 30
        quantize([[49.513,43,23],[49.513,43,23]], [0, 10, 20, 30, 45, 90])
        array([[45, 45, 20],
       [45, 45, 20]])

        x = np.linspace(0, 10, 41)
        y = np.linspace(0, 11, 11)
        z = my_quantize(x,y,mode='round')

        mu.plot_signal(x,format='o')
        mu.plot_signal(y,format='o')
        mu.plot_signal(z,format='o')
    Args:
        val        The value to quantize
        to_values  The allowed values
    Returns:
        Closest value among allowed values.
    """
    vals = np.asarray(vals)
    if mode == 'round':
        def single_quantize(x):
            diff = np.abs(x - np.asarray(to_values))
            mini = np.min(diff)
            idx = int(np.where(diff == mini)[0])
            return to_values[idx]
    elif mode == 'floor': # may give errors if it doesnt find any lower value
        def single_quantize(x):
            diff = x - np.asarray(to_values)
            idx = np.where(diff >= 0, diff, np.inf).argmin()
            return to_values[idx]
    elif mode == 'ceil': # may give errors if it doesnt find any higher value
        def single_quantize(x):
            diff = x - np.asarray(to_values)
            idx = np.where(diff <= 0, diff, -1*np.inf).argmax()
            return to_values[idx]
    foo = np.vectorize(single_quantize)
    return foo(vals)

def msqe(x,xq):
   return np.mean(np.square(x-xq))
   
def logEnergy(sig):
    sig2=np.square(sig) # Elevar al cuadrado las muestras de la senal
    sumsig2=np.sum(sig2) # Sumatoria
    logE=10*np.log10(sumsig2) # Convertir a dB
    dc=np.mean(sig)
    # Promedio
    return logE, dc

def transient_power(sig,db=False):
    if db:
        return 10 * np.log10(np.square(sig))
    else:
        return np.square(sig)

def mean_power(sig,db=False):
    transient_power = transient_power(sig,db=False)
    if db:
        return 10 * np.log10(np.mean(transient_power,axis=-1))
    else:
        return np.mean(transient_power,axis=-1)
