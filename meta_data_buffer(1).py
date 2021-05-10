from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
from fuzzyClassyficator import FuzzyClassyficator
from fuzzyClassyficator import Fuzzy_channel_choose

class MetaDataBuffer:
    def __init__(self, f, channels):
        self.signal_frequency = f
        self.channels = channels  # num of channels
        self.connector = Connector(self.signal_frequency, self.channels, self)
        self.chanel_trunc = []
        for i in range(0, channels):
            self.chanel_trunc.append(0)
        self._signals = []  # list of numpy array
        for i in range(0, channels):
            self._signals.append([])
        self.fuzzyClassifier = []
        for i in range(0, channels):
            self.fuzzyClassifier.append(FuzzyClassyficator(self.signal_frequency))

    def add_meta_data(self, channel, meta_data):
        meta_data.get_signal(self, channel)
        qrs_class, u, d = self.fuzzyClassifier[channel].classify(meta_data)
        self.connector.add_metadata(meta_data, qrs_class, u, d, channel)

    def add_sample(self, channel, sample):
        self._signals[channel].append(sample)
        if len(self._signals[channel]) > self.time(3000):
            self._signals[channel] = self._signals[channel][1:]
            self.chanel_trunc[channel] += 1

    def get_signal(self, qrs_ones, channel):
        return self._signals[channel][qrs_ones - self.chanel_trunc[channel]:]

    def get_signal_rr(self, R_center, sc_r_center, channel):
        return self._signals[channel][R_center - self.chanel_trunc[channel]:sc_r_center - self.chanel_trunc[channel]]

    def time(self, time):
        return int(time * self.signal_frequency / 1000.0 + 0.5)

    def get_last_nR_avg_height(self, n, channel):
        nR = self.connector.get_last_nR_height(n, channel)
        if len(nR) != 0:
            return sum(nR)/len(nR)
        else:
            return None

    def get_last_nR_min_height(self, n, channel):
        nR = self.connector.get_last_nR_height(n, channel)
        if nR:
            return min(nR)
        else:
            return None

    def get_last_R_pos(self, channel):
        # daj nam R które zawiera kanały za wyłączeniem obecnie rozpatrywanego
        nR = self.connector.get_last_R(channel)
        return nR

    def get_last_channel_R_pos(self, channel):
        nR = self.connector.get_last_channel_R_pos(channel)
        return nR
        # daj nam R który ma obecnie rozpatrywany kanał


# konektor ustala które QRS/kanał są tym samym wystąpieniem qrs
class Connector:
    def __init__(self, f, channels, buffer):
        self.signal_frequency = f
        self.channels = channels
        self.crosses = []
        self.sorted_crosses = []  # krosy posortowane po Rcenter po analizie st lub zgloszeniu niekomorowości
        # w celu poprawnego zapisu do pliku
        self.last_cross_ID = -2
        self.buffer = buffer
        self.channel_choose = Fuzzy_channel_choose(channels)

    def get_last_R(self, channel):
        if self.crosses:
            return self.crosses[-1].r_center
        return 0


    def get_last_channel_R_pos(self, channel):
        for cross in reversed(self.crosses):
            for channel_md in cross.channel_metadata:
                if channel == channel_md[0]:
                    return cross.r_center
        return 0

    def get_last_nR_height(self, n, channel):
        ret_list = []
        for cross in reversed(self.crosses):
            for channel_md in cross.channel_metadata:
                if channel == channel_md[0]:
                    ret_list.append(channel_md[1].R_height)
            if len(ret_list) >= n:
                break
        return ret_list

    def add_metadata(self, metadata, qrs_class, u, d, channel):
        if u != [1]:    # classified data
            self.last_cross_ID += self.channel_connect(channel, metadata, qrs_class, u, d, self.last_cross_ID + 2)  # new cross || crossID +1

            # dla każdego punktu łączenia który przekroczył licznosć kanałów == N
            # oraz jest odległy o trzy zespoły względem ostatno dodanego
            pause = 1  # liczba qrsów zapasu od frontu do sprawdzania st minimum 0
            for i in self.crosses:
                if self.crosses[-1].r_center > i.r_center >= self.crosses[-1].r_center - self.time(2000) - pause and i.size >= self.channels - 1:
                    i.check_st()

    def channel_connect(self, channel, metadata, qrs_class, u, d, counter):
        mt_r = metadata.r

        # przeszukanie punktów łączenia kanałów w celu dopasowania obecnie rozpatrywanego
        for i in range(len(self.crosses) - 1, 0, -1):
            if abs(mt_r - self.crosses[i].r_center) < self.time(50):
                self.crosses[i].add_channel(channel, metadata, qrs_class, u, d, counter)
                return 0

        # utworzenie nowego punktu łączenia kanałow
        self.crosses.append(MetaDataCross(self.signal_frequency, self.channels, self))
        self.crosses[-1].add_channel(channel, metadata, qrs_class, u, d, counter + 1)
        return 1

    # prośba! do connector o udostępnienie md innego łącznika
    def get_metadata(self, cross_id, channel):
        for i in range(len(self.crosses) - 1, 0, -1):
            if self.crosses[i].cross_ID == cross_id:
                sc_r_center = self.crosses[i].get_self_channel_metadata(channel)
                return sc_r_center
        return None

    # prośba! do connector o udostępnienie fragmentu sygnału RR
    def get_signal(self, cross_ID, channel):
        first_center = 0
        second_center = 0
        for i in range(len(self.crosses) - 1, 0, -1):
            if self.crosses[i].cross_ID == cross_ID + 1:
                for chan in self.crosses[i].channel_metadata:
                    if chan[0] == channel:
                        second_center = chan[1].r

        for i in range(len(self.crosses) - 1, 0, -1):
            if self.crosses[i].cross_ID == cross_ID:
                for chan in self.crosses[i].channel_metadata:
                    if chan[0] == channel:
                        first_center = chan[1].r
        #print(first_center)
        #print(second_center)
        return self.buffer.get_signal_rr(int(first_center), int(second_center), channel)

    def get_next_mdc_md_ud(self, cross_ID):
        for i in range(len(self.crosses) - 1, 0, -1):
            if self.crosses[i].cross_ID == cross_ID + 1:
                return self.crosses[i].get_self_metadata()

    def time(self, time):
        return int(time * self.signal_frequency / 1000.0 + 0.5)




# klasa używana do analizy własności QRS zsumowanych po kanale()
# przechowuje chanelx(MD + u,d)
class MetaDataCross:
    def __init__(self, f, channels, connector):
        self.channels = channels  # liczność kanałów
        self.connector = connector  # *parent
        self.cross_ID = None  # liczba porządkowa w kolejności wykrycia pierwszego qrs
        self.check = False  # ST SPRAWDZONE
        self.signal = None  # R-R Signal REF
        self.type = 0          # 1 = N --- 0 != N

        self.signal_frequency = f
        self.size = 0  # zajęte kanały
        self.channel_metadata = []  # array of md+ud [channsel]
        self.r_center = 0  # centrum qrs (R)
        self.best_channel = None

    def add_channel(self, channel, metadata, qrs_class, u, d, cross_ID):
        self.cross_ID = cross_ID

        evolution_type = False                                                  # true = verticular
        if np.argmax(u) == qrs_class:
            evolution_type = True
            self.type += 1

        self.r_center = (self.r_center * len(self.channel_metadata) + metadata.r) / (len(self.channel_metadata) + 1)        # Rcenter update
        self.channel_metadata.append([channel, metadata, evolution_type, u, d])                                                # index_metadata update
        self.size += 1

    # sprawdz poziom st i zapisz do pliku
    def check_st(self):
        if not self.check:
            self.check = True
            #
            # next_index_metadata = self.get_next_cross_metadata()  # index_metadata
            # if not next_index_metadata or len(next_index_metadata) < self.channels:                                   # 'TDOD' możliwe pominięcia - rozwioązanie- odpalanie ze zmniejzoną ilością kanałów w późniejszym czasie funkcję wyżej
            #     self.check = False
            #     return

            # if len(self.channel_metadata) < self.channels:                                   # 'TDOD' możliwe pominięcia - rozwioązanie- odpalanie ze zmniejzoną ilością kanałów w późniejszym czasie funkcję wyżej
            #     self.check = False
            #     return

            channel = self.connector.channel_choose.fuzzy_channel_choose(self.channel_metadata, self.channel_metadata)
            #channel = self.connector.channel_choose.fuzzy_channel_choose(self.channel_metadata, next_index_metadata)
            self.get_RR_signal(channel)
            self.filter()

            local_md = self.get_self_channel_metadata(channel)
            next_md = self.get_id_metadata(self.cross_ID + 1, channel)

            # ustalenie punktów charakterystycznych
            j_point = self.time(80)
            local_qrs_end = local_md.qrs_end - local_md.r
            local_qrs_ones = -1 * (next_md.r - next_md.qrs_ones )
            local_p_end = self.find_p_end(local_qrs_ones)

            # porównanie pq st
            ST = False
            try:
                if abs(abs(self.signal[j_point] - self.signal[local_qrs_end]) - abs(self.signal[local_p_end] - self.signal[local_qrs_ones])) > 0.4:
                    ST = True
            except:
                ST = False
                print("EXCEPT")
            #
            # plt.figure(2)
            # plt.clf()
            # plt.plot()
            # t = np.linspace(0, len(self.signal), len(self.signal), endpoint=False)
            # plt.plot(t, self.signal,
            #          j_point, self.signal[j_point], 'bp',
            #          local_qrs_end, self.signal[local_qrs_end], 'rp',
            #          len(self.signal)+local_p_end, self.signal[local_p_end], 'gp',
            #          len(self.signal)+local_qrs_ones, self.signal[local_qrs_ones], 'yp',
            #          len(self.signal)+q, self.signal[q], 'bp',)
            # plt.show()
            index = 0
            for id, md in enumerate(self.channel_metadata):
                if md[0] == channel:
                    index = id
            outfile = open("out.txt", 'a')
            outfile.write(str(int(self.channel_metadata[index][1].r))+' ' + str(int(ST)) + '\n')
            outfile.close()

    def get_next_cross_metadata(self):
        return self.connector.get_next_mdc_md_ud(self.cross_ID)

    def find_p_end(self, local_qrs_ones):
        # if local_qrs_ones - self.time(60) < len(self.signal):
        return local_qrs_ones - self.time(60)
        #return len(self.signal) - 1

    def get_id_metadata(self, Cross_id, channel):
        return self.connector.get_metadata(Cross_id, channel)

    def get_self_metadata(self):
        return self.channel_metadata

    def get_self_channel_metadata(self, channel):
        md = self.channel_metadata[0]
        #print(len(self.channel_metadata))
        for i in self.channel_metadata:
            if i[0] == channel:
                md = i
        return md[1]

    def get_RR_signal(self, channel):
        self.signal = self.connector.get_signal(self.cross_ID, channel)

    def time(self, time):
        return int(time * self.signal_frequency / 1000.0 + 0.5)

    def get_center(self):
        return self.r_center

    def filter(self):
        return 1


