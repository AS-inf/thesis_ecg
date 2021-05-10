from qrs_detector import QrsDetector
from meta_data_buffer import MetaDataBuffer
from fuzzy_threshold import FuzzyThreshold
import sys


class EcgReader:
    def __init__(self, f, channels, outfile):
        self.channels = channels
        self.signalFrequency = f
        self.qrs_detector = []
        self.mdb = MetaDataBuffer(self.signalFrequency, self.channels, outfile)
        self.fuzzyThreshold = FuzzyThreshold(self.signalFrequency, self.channels, self.mdb)

    def run(self):
        for channel in range(0, self.channels):
            self.qrs_detector.append(QrsDetector(self.signalFrequency, channel, self.mdb, self.fuzzyThreshold))

        while True:
            for text in sys.stdin:                          # collect data
                text2 = text.rstrip()[:-1]
                samples = text2.split('%')
                channel = 0
                for sample in samples:
                    if sample == 'D':
                        exit(0)
                    elif sample != '':
                        if channel < self.channels:
                            self.qrs_detector[channel].add_sample(float(sample))
                            self.mdb.add_sample(channel, float(sample))
                    else:
                        exit(1)
                    channel += 1
                break
