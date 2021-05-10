from __future__ import division, print_function
from scipy.signal import butter, lfilter
from metaDataInterface import MetaDataInterface
import numpy as np

from matplotlib import pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch
import copy


class QrsPreciseDetector:
    def __init__(self, f, window):
        self.cutoff_window = window
        self.signal_frequency = f
        self.signal = []
        self.q = 0
        self.r = 0
        self.s = 0
        self.qrs_ons = 0
        self.qrs_end = 0
        self.UPdirection = False
        self.Q_on_R = False
        self.rel_pos = 0

    def time(self, time):
        return int(time * self.signal_frequency / 1000.0 + 0.5)

    def relt(self, sample):
        if sample:
            return self.rel_pos + sample - self.cutoff_window
        return self.rel_pos - self.cutoff_window

    def analyze(self, signal, relation_pos):
        if len(signal) == 0:
            return None
        self.rel_pos = relation_pos
        self.signal = signal
        self.signal = self.filter()
        self.r = self.find_r()
        self.q = self.find_q()
        self.s = self.find_s()
        self.qrs_ons = self.find_qrs_ones()
        self.qrs_end = self.find_qrs_end()
        R_height = self.signal[self.r]

        t = np.linspace(0, len(self.signal), len(self.signal), endpoint=False)
        plt.figure()
        plt.clf()


        plt.plot(t, self.signal, label='Sygnał EKG')
        plt.plot(self.qrs_ons, self.signal[self.qrs_ons], 'rp'
                 , self.q, self.signal[self.q], 'rp'
                 , self.r, self.signal[self.r], 'rp'
                 , self.s, self.signal[self.s], 'rp'
                 , self.qrs_end, self.signal[self.qrs_end], 'rp'
                 )
        # # print(self.Q_on_R)
        plt.xlabel("Numer Próbki")
        plt.ylabel("Amplituda [mV]")
        plt.legend(loc="lower right")
        plt.show()
        #print(self.relt(self.r))
        return MetaDataInterface(self.relt(self.qrs_ons), self.relt(self.q), self.relt(self.r),  self.relt(self.s), self.relt(self.qrs_end), R_height)

    def filter(self):
        lowcut = 1
        highcut = 60
        self.signal = np.insert(self.signal, 0, [self.signal[0]] * 200)
        y = butter_bandpass_filter(self.signal, lowcut, highcut, self.signal_frequency, order=2)[200:]
        self.signal = self.signal[200:]
        return y

    def find_q(self):
        window = self.time(60)
        if self.r-window < 0:
            window = self.r
            print("WINDOW")
        if self.UPdirection:
            q = min(self.signal[self.r-window: self.r])
        else:
            q = max(self.signal[self.r-window: self.r])
        local_min = np.where(self.signal == q)[0][0]-1

        # diff = []
        # for i in range(local_min, self.r-1):
        #     diff.append((self.signal[i+1] - self.signal[i])**2)

        # max_loc = 0
        # for i in range(len(diff)-2, 0, -1):
        #     if diff[i] >= diff[i+1]:
        #         max_loc = i
        #     else:
        #         break
        # min_loc = 0
        # for i in range(max_loc-1, 0, -1):
        #     if diff[i] < diff[i+1]:
        #         min_loc = i
        #     else:
        #         break
        # max_loc2 = 0
        # for i in range(min_loc-1, 0, -1):
        #     if diff[i] > diff[i+1]:
        #         max_loc2 = i
        #     else:
        #         break
        #
        # if max_loc2 == 1 or max_loc2 == 0:
        #     return local_min
        #
        # if (diff[max_loc] + diff[max_loc2])/2 > 15 * diff[min_loc]:
        #     print(max_loc, max_loc2, min_loc)
        #     local_min = min_loc + local_min
        #     self.Q_on_R = True

        return local_min

    def find_s(self):
        window = self.time(70)
        if self.UPdirection:
            q = min(self.signal[self.r:window+self.r])
        else:
            q = max(self.signal[self.r:window + self.r])
        for i in range(self.r, window+self.r):
            if self.signal[i] == q:
                return i
        return 0

    def find_r(self):
        peak_pos = int(len(self.signal)/2)#self.time(200)
        if self.signal[peak_pos] > 0:
            self.UPdirection = True

        if self.UPdirection:
            while peak_pos < len(self.signal)-1 and peak_pos > 1:
                if self.signal[peak_pos-1] > self.signal[peak_pos]:
                    peak_pos -= 1

                    continue
                elif self.signal[peak_pos+1] > self.signal[peak_pos]:
                    peak_pos += 1

                    continue
                break
        else:
            while peak_pos < len(self.signal)-1 and peak_pos > 1:
                if self.signal[peak_pos - 1] < self.signal[peak_pos]:
                    peak_pos -= 1
                    continue
                elif self.signal[peak_pos + 1] < self.signal[peak_pos]:
                    peak_pos += 1
                    continue
                break
        return peak_pos

    def find_qrs_ones(self):
        # if self.Q_on_R:
        #     diff = []
        #     window = self.time(60)
        #     for i in range(self.q - window, self.q - 1):
        #         diff.append((self.signal[i + 1] - self.signal[i]) ** 2)
        #
        #     for i in range(len(diff)-1, 0, -1):
        #         if diff[i] < diff[-1]:
        #             return i + self.q - window -2
        #     return self.q - window
            # pochodna2 po sygnale < pochodna Q-on-r

        # go from right to left
        return self.q - self.time(25)
        # ones = self.q - self.time(10)
        # while ones > 0:
        #     #print("SUM: " + str(sum(self.signal[ones-1:self.q+1])))
        #     #print("AVG: " + str(((self.signal[ones]+self.signal[self.q])/2 * (self.q - ones))))
        #     Triangle_mass = sum(self.signal[ones:self.q]) - ((self.signal[ones]+self.signal[self.q])/2 * (self.q - ones))
        #     if Triangle_mass > 0.15:
        #         return ones - 2
        #     ones -= 1

    def find_qrs_end(self):
        window = self.time(80)  # 15ms)
        threshold = 4
        approximate_q = [self.s + self.time(100)]
        stop = 0
        i = 0

        while threshold >= 0.1:
            if not stop:
                for i in range(self.s + window, approximate_q[-1]):
                    if max(self.signal[i:i + window]) - min(self.signal[i:i + window]) > threshold:
                        # max po prawej od minimum
                        if np.where(self.signal[i:i + window] == max(self.signal[i:i + window])) > np.where(self.signal[i:i + window] == min(self.signal[i:i + window])):
                            if i != approximate_q[-1]:
                                approximate_q.append(i)
                                stop = i
                                break
                threshold /= 1.25
            if stop:
                for i in range(self.s + window, approximate_q[-1]):
                    if max(self.signal[i:i + window]) - min(self.signal[i:i + window]) > threshold:
                        # max po prawej od minimum
                        if np.where(self.signal[i:i + window] == max(self.signal[i:i + window])) > np.where(self.signal[i:i + window] == min(self.signal[i:i + window])):
                            if i != approximate_q[-1]:
                                approximate_q.append(i)
                                break
                threshold /= 1.25

        s = approximate_q[-1]
        if s >= len(self.signal):
            s = len(self.signal) - 1
        return s


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

