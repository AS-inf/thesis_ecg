import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class FuzzyThreshold:
    def __init__(self, frequency, channels, mdb):
        self.signal_frequencu = frequency
        self.channels = channels
        self.mdb = mdb
        self.sample_num = 0

        R_avg_SIG_avg = ctrl.Antecedent(np.arange(0, 1, 0.1), 'R_avg_SIG_avg')
        R_avg_SIG_avg['low'] = fuzz.trimf(R_avg_SIG_avg.universe, [0, 0.4, 0.4])
        R_avg_SIG_avg['med'] = fuzz.trimf(R_avg_SIG_avg.universe, [0.25, 0.5, 0.75])
        R_avg_SIG_avg['high'] = fuzz.trimf(R_avg_SIG_avg.universe, [0.6, 1, 1])

        R_min_SIG_avg = ctrl.Antecedent(np.arange(0, 1, 0.1), 'R_min_SIG_avg')
        R_min_SIG_avg['low'] = fuzz.trimf(R_min_SIG_avg.universe, [0, 0.4, 0.4])
        R_min_SIG_avg['med'] = fuzz.trimf(R_min_SIG_avg.universe, [0.25, 0.5, 0.75])
        R_min_SIG_avg['high'] = fuzz.trimf(R_min_SIG_avg.universe, [0.6, 1, 1])

        R_pos_R_channel_pos = ctrl.Antecedent(np.arange(0, 2*frequency, 1), 'R_pos_R_channel_pos')
        R_pos_R_channel_pos['Vlow'] = fuzz.trimf(R_pos_R_channel_pos.universe, [0, 0, 0.25*frequency])
        R_pos_R_channel_pos['low'] = fuzz.trimf(R_pos_R_channel_pos.universe, [0.2*frequency, 0.35*frequency, 0.45*frequency])
        R_pos_R_channel_pos['med'] = fuzz.trimf(R_pos_R_channel_pos.universe, [0.4*frequency, 0.6*frequency, 0.75*frequency])
        R_pos_R_channel_pos['high'] = fuzz.trimf(R_pos_R_channel_pos.universe, [0.65*frequency, 0.75*frequency, 0.85*frequency])
        R_pos_R_channel_pos['Vhigh'] = fuzz.trimf(R_pos_R_channel_pos.universe, [0.8*frequency, 2*frequency, 2*frequency])

        threshold_CNT = ctrl.Consequent(np.arange(0.1, 0.9, 0.01), 'threshold')
        #
        threshold_CNT['Vlow'] = fuzz.trimf(threshold_CNT.universe, [0.1, 0.1, 0.3])
        threshold_CNT['low'] = fuzz.trimf(threshold_CNT.universe, [0.2, 0.4, 0.5])
        threshold_CNT['med'] = fuzz.trimf(threshold_CNT.universe, [0.4, 0.55, 0.7])
        threshold_CNT['high'] = fuzz.trimf(threshold_CNT.universe, [0.6, 0.7, 0.8])
        threshold_CNT['Vhigh'] = fuzz.trimf(threshold_CNT.universe, [0.65, 0.9, 0.9])
        threshold_CNT.view()

        rule0 = ctrl.Rule(R_avg_SIG_avg['low'], threshold_CNT['Vlow'])
        rule1 = ctrl.Rule(R_avg_SIG_avg['med'], threshold_CNT['low'])
        rule2 = ctrl.Rule(R_avg_SIG_avg['high'] | R_min_SIG_avg['med'], threshold_CNT['high'])
        rule3 = ctrl.Rule(R_avg_SIG_avg['high'] | R_min_SIG_avg['high'], threshold_CNT['Vhigh'])

        rule4 = ctrl.Rule(R_min_SIG_avg['low'] | R_avg_SIG_avg['med'], threshold_CNT['low'])
        rule5 = ctrl.Rule(R_min_SIG_avg['low'] | R_avg_SIG_avg['high'], threshold_CNT['med'])
        rule6 = ctrl.Rule(R_min_SIG_avg['med'] | R_avg_SIG_avg['med'], threshold_CNT['med'])
        rule7 = ctrl.Rule(R_min_SIG_avg['med'] | R_avg_SIG_avg['high'], threshold_CNT['high'])
        rule8 = ctrl.Rule(R_min_SIG_avg['high'], threshold_CNT['Vhigh'])

        rule9 = ctrl.Rule(R_pos_R_channel_pos['Vlow'], threshold_CNT['Vhigh'])
        rule10 = ctrl.Rule(R_pos_R_channel_pos['low'], threshold_CNT['high'])
        rule11 = ctrl.Rule(R_pos_R_channel_pos['med'], threshold_CNT['med'])
        rule12 = ctrl.Rule(R_pos_R_channel_pos['high'], threshold_CNT['low'])
        rule13 = ctrl.Rule(R_pos_R_channel_pos['Vhigh'], threshold_CNT['Vlow'])

        tipping_ctrl = ctrl.ControlSystem([rule0, rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13])
        self.tipping = ctrl.ControlSystemSimulation(tipping_ctrl)

    def get_channel_threshold(self, channel, fuzzy_signal):
        # SKALOWANIE!!!!!!!!!!!!!!!
        self.sample_num += 1
        signal_avg = sum(fuzzy_signal)/len(fuzzy_signal)
        #print("signal_avg: ", signal_avg)
        n = 10
        R_avg = self.mdb.get_last_nR_avg_height(n, channel)
        if R_avg is None:
            R_avg = signal_avg
        #print("R_avg: ", R_avg)
        n = 20
        R_min = self.mdb.get_last_nR_min_height(n, channel)
        if R_min is None:
            R_min = signal_avg
        #print("R_min: ", R_min)
        R_pos = self.mdb.get_last_R_pos(channel)
        #print("R_pos: ", R_pos)
        R_channel_pos = self.mdb.get_last_channel_R_pos(channel)
        #print("R_channel_pos: ", R_channel_pos)

        self.tipping.input['R_avg_SIG_avg'] = abs(R_avg - signal_avg)
        self.tipping.input['R_min_SIG_avg'] = abs(R_min - signal_avg)
        self.tipping.input['R_pos_R_channel_pos'] = abs(R_pos - R_channel_pos)
        self.tipping.compute()
        #print("FUZZY: ", signal_avg + self.tipping.output['threshold'] * abs(R_avg - signal_avg))

        #print("POS80: ", fuzzy_signal[int(len(fuzzy_signal) * 80 / 100)])

        return signal_avg + self.tipping.output['threshold'] * abs(R_avg - signal_avg)

