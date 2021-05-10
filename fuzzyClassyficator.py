import skfuzzy as fuzz
import numpy as np
from matplotlib import pyplot as plt

from skfuzzy import control as ctrl


class FuzzyClassyficator:
    def __init__(self, f):
        self.signal_frequency = f
        self.i = 0
        self.mem = [[], [], [], []]
        self.center = None
        self.u = None
        self.data = None

    def classify(self, metadata):
        colors = ['b', 'r', 'g', 'y']
        sf = metadata.get_selected_features()
        self.mem[0].append(sf[0])
        self.mem[1].append(sf[1])
        self.mem[2].append(sf[2])
        self.mem[3].append(sf[3])
        self.i += 1
        if self.i >= 50:

            self.mem[0] = self.mem[0][1:]
            self.mem[1] = self.mem[1][1:]
            self.mem[2] = self.mem[2][1:]
            self.mem[3] = self.mem[3][1:]

            self.data = np. vstack((self.mem[0], self.mem[1], self.mem[2], self.mem[3]))
            n_centers = 2
            self.center, self.u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(self.data, n_centers, 2, error=0.005, maxiter=1000, init=self.u)

            # cluster_membership = np.argmax(self.u, axis=0)
            # fig1, axes = plt.subplots(1, 1)
            # for j in range(n_centers):
            #     axes.plot(self.data[0, cluster_membership == j], self.data[1, cluster_membership == j], '.', color=colors[j])
            #
            # for pt in self.center:
            #     axes.plot(pt[0], pt[1], 'rs')
            # axes.set_title('Centers = {0}; FPC = {1:.2f}'.format(n_centers, fpc))
            # axes.axis('off')
            # fig1.tight_layout()
            # plt.show()

            clas_id = self.class_identyfication(d)
            return clas_id,  [self.u[0][-1], self.u[1][-1]], [d[0][-1], d[1][-1]]
        return 0, [1], [0]   # nie klasyfikowane sprzedajemy jako zatokowe

    def class_identyfication(self, d):
        # należy do klasy z naj=większą linzisdsjfą
        classes = [0, 0]
        cluster_membership = np.argmax(self.u, axis=0)
        for i in cluster_membership:
            classes[i] += 1
        #print(classes)
        #print(np.argmax(classes))
        return np.argmax(classes)


class Fuzzy_channel_choose:
    def __init__(self, channels):
        self.channels = channels
        this_u = ctrl.Antecedent(np.arange(0, 1, 0.01), 'this_u')
        this_u['low'] = fuzz.trimf(this_u.universe, [0, 0.5, 1])
        this_u['high'] = fuzz.trimf(this_u.universe, [0, 1, 1])

        next_u = ctrl.Antecedent(np.arange(0, 1, 0.01), 'next_u')
        next_u['low'] = fuzz.trimf(next_u.universe, [0, 0.5, 1])
        next_u['high'] = fuzz.trimf(next_u.universe, [0, 1, 1])

        chanel_goodnes = ctrl.Consequent(np.arange(0, 100, 1), 'goodnes')
        chanel_goodnes['low'] = fuzz.trimf(chanel_goodnes.universe, [0, 0, 50])
        chanel_goodnes['med'] = fuzz.trimf(chanel_goodnes.universe, [25, 50, 75])
        chanel_goodnes['high'] = fuzz.trimf(chanel_goodnes.universe, [50, 100, 100])

        rule0 = ctrl.Rule(this_u['low'] | next_u['low'], chanel_goodnes['low'])
        rule1 = ctrl.Rule(this_u['high'] | next_u['low'], chanel_goodnes['med'])

        rule2 = ctrl.Rule(this_u['low'] | next_u['high'], chanel_goodnes['med'])
        rule3 = ctrl.Rule(this_u['high'] | next_u['high'], chanel_goodnes['high'])

        tipping_ctrl = ctrl.ControlSystem([rule0, rule1, rule2, rule3])
        self.tipping = ctrl.ControlSystemSimulation(tipping_ctrl)

    def fuzzy_channel_choose(self, this_index_metadata, next_index_metadata):

        this_u_array = np.zeros((self.channels, 2))
        next_u_array = np.zeros((self.channels, 2))

        # |N|V| order by chanel
        for channel in this_index_metadata:
            #print(channel[2])
            if channel[3][0] > channel[3][1]:
                this_u_array[channel[0]][0] = channel[3][0]
                this_u_array[channel[0]][1] = channel[3][1]
            else:
                this_u_array[channel[0]][0] = channel[3][1]
                this_u_array[channel[0]][1] = channel[3][0]

        # |N|V| order by chanel
        for channel in next_index_metadata:
            #print(channel[2])
            if channel[3][0] > channel[3][1]:
                next_u_array[channel[0]][0] = channel[3][0]
                next_u_array[channel[0]][1] = channel[3][1]
            else:
                next_u_array[channel[0]][0] = channel[3][1]
                next_u_array[channel[0]][1] = channel[3][0]

        scores = np.zeros(self.channels)
        for index, channel in enumerate(this_u_array):
            self.tipping.input['this_u'] = this_u_array[index][0]
            self.tipping.input['next_u'] = next_u_array[index][0]
            self.tipping.compute()
            scores[index] = self.tipping.output['goodnes']
        return np.where(scores == max(scores))[0][0]
