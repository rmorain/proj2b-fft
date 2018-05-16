import numpy as np


def convolve(signal, filter):
    """
    Problem Formulation
    -------------------
    Zero pad the signal and the filter
    a, b = preprocess(a, b)

    Convert signal to frequency domain
    A = FFT(a)

    Convert filter to frequency domain
    B = FFT(b)

    Multiplication in the frequency domain is convolution in the time domain
    Y = A*B  # Element-wise multiplication

    Return the output to the time domain
    y = INVERSE_FFT(Y)

    Remove the zero padding
    return postprocess(y)

    Parameters
    ----------
    signal : np.array (length n)
    filter : np.array (length m)

    Returns
    -------
    convolved : np.array (length n)

    Complexity
    ----------
    The highest order step is the inverse fft so this function has a complexity of O(N^2)
    """

    # Make the signal and filter the correct size
    padded_signal, padded_filter = preprocess(signal, filter)   # Constant time
    fft_signal = fft(padded_signal) # Log(n) complexity
    fft_filter = fft(padded_filter) # Log(n) complexity
    filtered_signal = np.multiply(fft_signal, fft_filter)   # Element wise multiply (p multiplies)
    time_signal = inverse_fft(filtered_signal)  # O(N^2)
    # Remove excess zeros
    time_signal = postprocess(time_signal, signal.size, filter.size)    # O(N)
    print("Done Filtering")
    # return np.convolve(filter, signal)  # Replace with your fft implementation
    return time_signal


def fft(signal):
    """
    Problem Formulation
    -------------------
    Divide the signal into even and odd components
    Run recursive fft on the even and odd components
    Base case where you return a single remaining element
    Recombine even and odd signals multiplied by complex kernels

    Parameters
    ----------
    signal : np.array (length n) (could be a filter also)

    Returns
    -------
    Frequency domain signal/filter

    Complexity
    ----------
    This function applies a divide and conquer technique.
    We have two log(n) functions being called at each recursive layer.
    The concatenation of the even part and the odd part is called every time fft is called so 2*log(n) times
    The complexity of this function is O(log(n))


    """
    if signal.size == 1:
        return signal

    even_part = fft(signal[::2])   # Only grab even elements
    odd_part = fft(signal[1::2])    # Only grab odd elements

    factor = np.exp(-2j * np.pi * np.arange(signal.size) / signal.size)
    return np.concatenate([even_part + factor[:int(signal.size / 2)] * odd_part,
                    even_part + factor[int(signal.size / 2):] * odd_part])


def inverse_fft(signal):
    """
    Problem Formulation
    -------------------
    Y1 = a copy of Y with all of the values replaced with their conjugates
    y1 = FFT(Y)
    y2 = a copy of y1 with all of the value replaced with their conjugates
    return y2 / n

    Parameters
    ----------
    signal : np.array (length n)

    Returns
    -------
    Time domain signal/filter

    Complexity
    ----------
    The most complex operation is the division so this function is O(N^2)
    """
    Y1 = np.conj(signal)    # Conjugates each element in signal which has p elements
    y1 = fft(Y1)    # Log(n)
    y2 = np.conj(y1)    # O(N)
    return np.divide(y2, signal.size)   # O(N^2)

def preprocess(a, b):
    """
    m = len(a)
    n = len(b)
    p1 = m + n - 1  # The maximum length of our answer
    find p such that p >= p1 is a power of 2
    pad a and b to length p
    return padded arrays

    Complexity
    ----------
    Nothing in this function is dependent on N other than the size of the array that
    we are appending to a and b.
    We will assume numpy can do this constant time.
    """
    m = len(a)
    n = len(b)
    p1 = m + n - 1 # The maximum length of the filtered signal
    # Find p such that p >= p1 is a power of 2
    p = int(np.power(2, np.ceil(np.log2(p1))))
    a = np.append(a, np.zeros(p - a.size))
    b = np.append(b, np.zeros(p - b.size))
    return a, b


def postprocess(signal, m, n):
    """
    Removes padded zeros
    Complexity
    ----------
    O(N)
    """
    return np.real(signal[0:m + n - 1]) # Removes complex part for each element p = m + n -1
