from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter


class MetaDataInterface:
    def __init__(self, qrs_ones, q, r, s, qrs_end, R_height):
        self.signal_frequency = 0
        self.qrs_ones = qrs_ones
        self.q = q
        self.r = r
        self.s = s
        self.qrs_end = qrs_end
        self.R_height = R_height
        self.signal = None

    def get_selected_features(self):
        self.signal = self.filter()
        qrs_time = self.get_qrs_time()
        VS = self.get_vs()
        PA = self.get_positive_area()
        NA = self.get_negative_area()
        self.signal = None #############################################################################################
        return [qrs_time, VS, NA, PA]


    def filter(self):
        lowcut = 0.7
        highcut = 17
        self.signal = np.insert(self.signal, 0, [self.signal[0]] * 200)

        y = butter_bandpass_filter(self.signal, lowcut, highcut, self.signal_frequency, order=3)[200:]
        self.signal = self.signal[200:]
        # t = np.linspace(0, len(self.signal), len(self.signal), endpoint=False)
        # plt.figure(1)
        # plt.clf()
        # plt.plot(t, self.signal, t, y,  self.q-self.qrs_ones, self.signal[self.q - self.qrs_ones], 'bp',
        #          self.r-self.qrs_ones, self.signal[self.r - self.qrs_ones], 'rp',
        #          self.s-self.qrs_ones, self.signal[self.s - self.qrs_ones], 'gp',
        #          self.qrs_end-self.qrs_ones, self.signal[self.qrs_end - self.qrs_ones], 'mp')
        # plt.show()
        return y

    def get_negative_area(self):
        start = self.r - self.qrs_ones
        #print(self.qrs_end - self.qrs_ones)
        end = min([self.qrs_end - self.qrs_ones, len(self.signal)])
        for i in range(self.r - self.qrs_ones, end):
            if self.signal[i] > 0 and start == self.r - self.qrs_ones:
                continue
            elif start == self.r - self.qrs_ones:
                start = i

        _sum = abs(sum(self.signal[start: end]))
        return _sum

    def get_positive_area(self):
        start = self.s - self.qrs_ones
        for i in range(self.s - self.qrs_ones, self.q - self.qrs_ones, -1):
            if self.signal[i] > 0 and start == self.s - self.qrs_ones:
                continue
            elif start == self.s - self.qrs_ones:
                start = i
        _sum = abs(sum(self.signal[:start]))
        return _sum

    def get_vs(self):
        avg = 0
        min = self.get_qrs_time()-2
        if len(self.signal) - 1 < min:
            min = len(self.signal) - 1
        for i in range(0, min):
            avg += abs(self.signal[i+1]-self.signal[i])
        return avg

    def get_qrs_time(self):
        return self.qrs_end - self.qrs_ones

    def get_signal(self, buffer, channel):
        self.signal_frequency = buffer.signal_frequency
        self.signal = buffer.get_signal(self.qrs_ones, channel)

    def show(self):
        t = np.linspace(0, len(self.signal), len(self.signal), endpoint=False)
        plt.figure(1)
        plt.clf()
        plt.plot(t, self.signal,  self.q-self.qrs_ones, self.signal[self.q - self.qrs_ones], 'bp',
                 self.r-self.qrs_ones, self.signal[self.r - self.qrs_ones], 'rp',
                 self.s-self.qrs_ones, self.signal[self.s - self.qrs_ones], 'gp',
                 self.qrs_end-self.qrs_ones, self.signal[self.qrs_end - self.qrs_ones], 'mp')
        plt.show()



def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

