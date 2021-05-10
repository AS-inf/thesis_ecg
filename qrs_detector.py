from qrs_precise_detection import QrsPreciseDetector
from copy import copy
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl



class QrsDetector:
    def __init__(self, f, channel, meta_data_buffer, fuzzyThreshold):
        self.signalFrequency = f
        self.signal = []  # queue.Queue(maxsize=4*self.signalFrequency)
        self.differ = [0]
        self.differ2 = [0]
        self.integer = [0]
        self.PIDwindow = self.time(116)
        self.prc_detector = QrsPreciseDetector(self.signalFrequency, int(self.PIDwindow/2)+self.time(300))
        self.mdb = meta_data_buffer
        self.channel = channel
        self.threshold_send_sample = []  # if last_sample_num = <- [1] then send <-[0] - [1]
        self.progress = -1
        self.diff_range = 0
        self.truncated = self.time(200)
        self.threshold = 0
        self.fuzzyThreshold = fuzzyThreshold


    def add_sample(self, sample):
        self.signal.append(sample)
            # qrs_window[3] -= 1
        self.sample_analyze()

    def sample_analyze(self):
        self.progress += 1
        if len(self.signal) > self.time(2000):
            self.signal = self.signal[1:]
            for qrs_window in self.threshold_send_sample:
                qrs_window[0] -= 1
                qrs_window[1] -= 1

        if len(self.signal) > 2:
            self.differ.append((self.signal[-3] - self.signal[-1]) * 1)
            while len(self.differ) > self.time(2000) - self.PIDwindow/2:
                self.truncated += 1
                self.differ = self.differ[1:]

        self.integral()
        self.differential()
        self.precise_detection()

        if self.progress > self.time(2000):
            self.check_threshold()
        else:
            self.truncated = 0
        return

    def precise_detection(self):
        if self.threshold_send_sample:
            if self.threshold_send_sample[0][2] <= self.progress:
                metadata = self.prc_detector.analyze(self.signal[self.threshold_send_sample[0][0]:self.threshold_send_sample[0][1]], self.threshold_send_sample[0][3])
                if metadata:
                    self.mdb.add_meta_data(self.channel, metadata)
                self.threshold_send_sample = self.threshold_send_sample[1:]

    def check_threshold(self):
        # FUZZY - siły sygnału długości o podwyższonej mocy
        fuzzy_signal = self.differ2[:-1]
        if self.diff_range == 0:
            self.threshold = self.fuzzy_threshold(fuzzy_signal)
        if self.differ2[-1] >= self.threshold:
            self.diff_range += 1
        elif self.diff_range > 3:  # minimal peak width
            diff_max = self.find_reverse_max(self.diff_range)
            self.threshold_send_sample.append([diff_max - self.time(300), diff_max + self.time(300), self.progress + self.time(400), self.truncated+diff_max])
            self.diff_range = 0

    def integral(self):
        if len(self.differ) > self.PIDwindow:
            while len(self.integer) < self.PIDwindow / 2:
                self.integer.append(self.integer[-1])
            self.integer.append(sum(power(self.differ[-self.PIDwindow:])))
            while len(self.integer) > self.time(2000) - self.PIDwindow / 2:
                self.integer = self.integer[1:]

    def differential(self):
        if len(self.integer) > self.PIDwindow:
            while len(self.differ2) < self.PIDwindow / 2:
                self.differ2.append(self.differ2[-1])
            self.differ2.append(sum(self.integer[-self.PIDwindow:]) / self.PIDwindow)
            while len(self.differ2) > self.time(2000) - self.PIDwindow:
                self.differ2 = self.differ2[1:]

    def find_reverse_max(self, max_range):
        return self.differ2.index(max(self.differ2[-1:-max_range:-1]))

    def time(self, time):
        return int(time * self.signalFrequency / 1000.0 + 0.5)

    def fuzzy_threshold(self, fuzzy_signal):
        return self.fuzzyThreshold.get_channel_threshold(self.channel, fuzzy_signal)

def power(list):
    return [x ** 2 for x in list]
