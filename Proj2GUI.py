import os
import signal
import sys
import time

import pyaudio
import wave
import numpy as np
from scipy.io.wavfile import read as read_wav, write as write_wav

from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout
from PyQt5.QtWidgets import QLabel, QPushButton, QLineEdit

from fft import convolve

CHUNK = 1024


class Proj2GUI(QWidget):
    """GUI to run project 2. Allows users to chose audio signals to convolve,
    and then hit a button to run the convolution. Then it allows the user
    to save the result to disk.
    """

    def __init__(self):
        super().__init__()
        self.default_text = (
            'Signal is the base audo file, and filter is the filter to apply '
            'to the audio file.'
        )
        self.save_loc = None

        self.initUI()

    def initUI(self):
        """Initializes the User Interface"""
        self.setWindowTitle('Convolution through FFT')
        self.setWindowIcon(QIcon('icon312.png'))

        vbox = QVBoxLayout()
        self.setLayout(vbox)

        self.signal_loc = QLineEdit('')
        self.filter_loc = QLineEdit('')
        self.load_signal = QPushButton('Load')
        self.load_filter = QPushButton('Load')
        self.play_signal = QPushButton('Play')
        self.play_filter = QPushButton('Play')
        self.convolve = QPushButton('Run Convolution')
        self.play_convolution = QPushButton('Play Results')
        self.output = QLabel(self.default_text)
        self.output.setMinimumSize(250, 0)

        # Signal
        h = QHBoxLayout()
        h.addWidget(QLabel('Signal File Location: '))
        h.addWidget(self.signal_loc)
        h.addWidget(self.load_signal)
        h.addWidget(self.play_signal)
        vbox.addLayout(h)

        # Filter
        h = QHBoxLayout()
        h.addWidget(QLabel('Filter File Location: '))
        h.addWidget(self.filter_loc)
        h.addWidget(self.load_filter)
        h.addWidget(self.play_filter)
        vbox.addLayout(h)

        # Output
        h = QHBoxLayout()
        h.addWidget(self.output)
        vbox.addLayout(h)

        # Convolution
        h = QHBoxLayout()
        h.addWidget(self.convolve)
        h.addWidget(self.play_convolution)
        vbox.addLayout(h)

        # Button functionality
        self.load_signal.clicked.connect(self.handleSignalFileSelect)
        self.load_filter.clicked.connect(self.handleFilterFileSelect)
        self.play_signal.clicked.connect(self.handlePlaySignal)
        self.play_filter.clicked.connect(self.handlePlayFilter)
        self.convolve.clicked.connect(self.handleConvolve)
        self.play_convolution.clicked.connect(self.handlePlayConvolution)

        self.show()

    def handleConvolve(self):
        """Runs the convolution when the convolve button is clicked.
        """
        def _getFile(widget, name):
            filename = widget.text()
            if filename == '':
                raise Exception((
                    'Cannot run convolution: must specify the {}'
                ).format(name))
            if not os.path.exists:
                raise Exception((
                    'Cannot run convolution: path specified {} does not exist'
                ).format(name))

            rate, data = read_wav(filename)
            data = pcm2float(data, 'float32')

            if len(data.shape) == 1:
                data = data.reshape((len(data), 1))

            return data, rate

        try:
            signal, rate = _getFile(self.signal_loc, 'signal')
            filter, _ = _getFile(self.filter_loc, 'filter')

            print('Signal has {} channel(s).'.format(signal.shape[1]))
            print('Filter has {} channel(s).'.format(signal.shape[1]))

            rs = None
            count = 0
            for i in range(signal.shape[1]):
                for j in range(filter.shape[1]):
                    convolution = float2pcm(
                        convolve(signal[:, i], filter[:, j]),
                        'int16'
                    )
                    convolution = convolution.reshape((len(convolution), 1))
                    if rs is None:
                        rs = convolution
                    else:
                        rs = np.append(rs, convolution, axis=1)
                    count += 1
            filename = self.saveFileDialog()
            if not filename:
                raise Exception('Error: Save file not specified')

            write_wav(filename, rate, convolution)
            self.save_loc = filename

        except Exception as e:
            self.output.setText(str(e))

    def handleSignalFileSelect(self):
        """Handles the selection of the signal file by setting its value
        in the signal line edit box.
        """
        filename = self.openFileNameDialog()
        if filename:
            self.signal_loc.setText(filename)

    def handleFilterFileSelect(self):
        """Handles the selection of the filter file by setting its value
        in the signal line edit box.
        """
        filename = self.openFileNameDialog()
        if filename:
            self.filter_loc.setText(filename)

    def handlePlayConvolution(self):
        """Plays the file saved by the most recent convolution.
        """
        if self.save_loc is None:
            self.output.setText('Error: need to run a convolution first')
        else:
            filename = self.save_loc
            self.play(filename, 'convolution')

    def handlePlaySignal(self):
        """Plays the file specified by in the signal line edit box.
        """
        filename = self.signal_loc.text()
        self.play(filename, 'signal')

    def handlePlayFilter(self):
        """Plays the file specified by in the filter line edit box.
        """
        filename = self.filter_loc.text()
        self.play(filename, 'filter')

    def play(self, filename, title):
        """Plays the audio file specified by filename.

        Parameters
        ----------
        filename : str
            The file to play.
        title : str
            The widget calling play. Used for messages.
        """
        if filename == '':
            self.output.setText('Cannot play {}: no file specified'.format(
                title
            ))
            return
        if not os.path.isfile(filename):
            self.output.setText('Cannot play {}: file does not exist'.format(
                title
            ))
            return

        self.output.setText('Playing {}'.format(title))
        print('Playing audio file')

        wf = wave.open(filename, 'rb')

        # instantiate PyAudio (1)
        p = pyaudio.PyAudio()

        # define callback (2)
        def callback(in_data, frame_count, time_info, status):
            data = wf.readframes(frame_count)
            return (data, pyaudio.paContinue)

        # open stream using callback (3)
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True,
                        stream_callback=callback)

        # start the stream (4)
        stream.start_stream()

        # wait for stream to finish (5)
        while stream.is_active():
            time.sleep(0.1)

        # stop stream (6)
        stream.stop_stream()
        stream.close()
        wf.close()

        print('\tDone playing')

        self.output.setText(self.default_text)

    def openFileNameDialog(self):
        """Opens an open file dialogue.
        """
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(
            self,
            'QFileDialog.getOpenFileName()',
            '',
            'Wav Files (*.wav)',
            options=options
        )
        return fileName

    def saveFileDialog(self):
        """Opens a save file dialogue.
        """
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(
            self,
            'QFileDialog.getSaveFileName()',
            '',
            'Wav Files (*.wav)',
            options=options
        )
        return fileName


def pcm2float(sig, dtype='float64'):
    """Convert PCM signal to floating point with a range from -1 to 1.

    Use dtype='float32' for single precision.

    Parameters
    ----------
    sig : array_like
        Input array, must have integral type.
    dtype : data type, optional
        Desired (floating point) data type.

    Returns
    -------
    numpy.ndarray
        Normalized floating point data.

    See Also
    --------
    float2pcm, dtype

    """
    sig = np.asarray(sig)
    if sig.dtype.kind not in 'iu':
        raise TypeError("'sig' must be an array of integers")
    dtype = np.dtype(dtype)
    if dtype.kind != 'f':
        raise TypeError("'dtype' must be a floating point type")

    i = np.iinfo(sig.dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig.astype(dtype) - offset) / abs_max


def float2pcm(sig, dtype='int16'):
    """Convert floating point signal with a range from -1 to 1 to PCM.

    Any signal values outside the interval [-1.0, 1.0) are clipped.
    No dithering is used.

    Note that there are different possibilities for scaling floating
    point numbers to PCM numbers, this function implements just one of
    them.  For an overview of alternatives see
    http://blog.bjornroche.com/2009/12/int-float-int-its-jungle-out-there.html

    Source: https://nbviewer.jupyter.org/github/mgeier/python-audio/blob/
              master/audio-files/audio-files-with-scipy-io.ipynb

    Parameters
    ----------
    sig : array_like
        Input array, must have floating point type.
    dtype : data type, optional
        Desired (integer) data type.

    Returns
    -------
    numpy.ndarray
        Integer data, scaled and clipped to the range of the given
        *dtype*.

    See Also
    --------
    pcm2float, dtype

    """
    sig = np.asarray(sig)
    if sig.dtype.kind != 'f':
        raise TypeError("'sig' must be a float array")
    dtype = np.dtype(dtype)
    if dtype.kind not in 'iu':
        raise TypeError("'dtype' must be an integer type")

    i = np.iinfo(dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig * abs_max + offset).clip(i.min, i.max).astype(dtype)


def pcm24to32(data, channels=1, normalize=True):
    """Convert 24-bit PCM data to 32-bit.

    Source: https://nbviewer.jupyter.org/github/mgeier/python-audio/blob/
              master/audio-files/audio-files-with-scipy-io.ipynb

    Parameters
    ----------
    data : buffer
        A buffer object where each group of 3 bytes represents one
        little-endian 24-bit value.
    channels : int, optional
        Number of channels, by default 1.
    normalize : bool, optional
        If ``True`` (the default) the additional zero-byte is added as
        least significant byte, effectively multiplying each value by
        256, which leads to the maximum 24-bit value being mapped to the
        maximum 32-bit value.  If ``False``, the zero-byte is added as
        most significant byte and the values are not changed.

    Returns
    -------
    numpy.ndarray
        The content of *data* converted to an *int32* array, where each
        value was padded with zero-bits in the least significant byte
        (``normalize=True``) or in the most significant byte
        (``normalize=False``).

    """
    if len(data) % 3 != 0:
        raise ValueError('Size of data must be a multiple of 3 bytes')

    out = np.zeros(len(data) // 3, dtype='<i4')
    out.shape = -1, channels
    temp = out.view('uint8').reshape(-1, 4)
    if normalize:
        # write to last 3 columns, leave LSB at zero
        columns = slice(1, None)
    else:
        # write to first 3 columns, leave MSB at zero
        columns = slice(None, -1)
    temp[:, columns] = np.frombuffer(data, dtype='uint8').reshape(-1, 3)
    return out


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    app = QApplication(sys.argv)
    w = Proj2GUI()
    sys.exit(app.exec())
