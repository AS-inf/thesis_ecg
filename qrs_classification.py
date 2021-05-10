import copy
from collections import namedtuple

HEART_MIN_FREQ = 10
HEART_MIN_DYNAMIC = 10

class FuzzyDataBase:
    rules = []

    def add_rule(self, premises, proposals):
        true_flag = True
        for premise in premises:
            if premise == 0:
                true_flag = False
        #if true_flag:
            #all(proposals) = max/min(premises)

class LinguisticVariable:
    def list_vars(self):
        variables = []
        for var in vars(self):
            variables.append(var)
        return variables

    def list_nz_vars(self):
        variables = []
        for var in vars(self):
            if eval('self.' + var) != 0:
                variables.append(var)
        return variables

    def set_var(self, var, val):
        exec('self.' + var + '=' + str(val))



class QrsLen(LinguisticVariable):
    def __init__(self):
        self.no_set = 0
        self.minimal = 0
        self.avg = 0
        self.maximal = 0
        self.top_out_of_range = 0            # 0/100 non fuzzy value
        self.bottom_out_of_range = 0         # 0/100 non fuzzy value

    def fuzzifification(self, qrs_len):

        print(self.list_vars())
        print(self.list_nz_vars())
        self.set_var("avg", 7)
        print(self.list_nz_vars())

        if qrs_len == 0:
            self.no_set = 100
            return
        if qrs_len < HEART_MIN_FREQ:
            self.bottom_out_of_range = 100
            return

        if 1 <= qrs_len <= 11:
            self.minimal = 3
            self.avg = 2



class QrsDynamic(LinguisticVariable):
    def __init__(self):
        self.no_set = 0
        self.minimal = 0
        self.avg = 0
        self.maximal = 0
        self.top_out_of_range = 0            # 0/100 non fuzzy value
        self.bottom_out_of_range = 0         # 0/100 non fuzzy value

    def fuzzifification(self, qrs_len):

        print(self.list_vars())
        print(self.list_nz_vars())
        self.set_var("avg", 7)
        print(self.list_nz_vars())

        if qrs_len == 0:
            self.no_set = 100
            return
        if qrs_len < HEART_MIN_FREQ:
            self.bottom_out_of_range = 100
            return
        if 1 <= qrs_len <= 11:
            self.minimal = 1
            self.avg = 5


class QrsClassifier:
    def __init__(self, f, qrs_signal):
        self.signal_frequency = f
        self.qrs_signal = copy.copy(qrs_signal)

        self.qrs_len = 0
        self.lin_qrs_len = QrsLen()
        self.lin_qrs_dynamic = QrsDynamic()




    def classification(self):
        print("classification")
        self.fuzzification()
        self.agregation()

    def fuzzification(self):
        self.lin_qrs_len.fuzzifification(10)
        self.lin_qrs_dynamic.fuzzifification(10)

        print("fuzzification")

    def agregation(self):
        #for all przesłanka do check wniosek

        #lookback

        print("Matching")

    def defuzzification(self):
        print("defuzzification")
