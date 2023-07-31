import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


def print_hi(name):
    print("Mencoba ANFIS")

    # Variabel input //niai minimum, nilai maksimum, jumlah poin//
    temperature = ctrl.Antecedent(np.linspace(0, 100, 101), 'temperature')

    #definisi membership function
    temperature['cold'] = fuzz.trimf(temperature.universe, [0, 0, 50]) # //nilai awal sebelum naik, nilai naik puncak, nilai menurun
    temperature['hot'] = fuzz.trimf(temperature.universe, [50, 100, 100])

    print(f"sebuah test: {temperature['cold']}")


    # Variable output
    fan_speed = ctrl.Consequent(np.linspace(0, 100, 101), 'fan_speed')
    fan_speed['slow'] = fuzz.trimf(fan_speed.universe, [0, 0, 50])
    fan_speed['fast'] = fuzz.trimf(fan_speed.universe, [50, 100, 100])

    # Rule
    rule1 = ctrl.Rule(temperature['cold'], fan_speed['slow'])
    rule2 = ctrl.Rule(temperature['hot'], fan_speed['fast'])

    # Control System
    fan_control = ctrl.ControlSystem([rule1, rule2])

    # Simulate ANFIS
    fan_simulation = ctrl.ControlSystemSimulation(fan_control)
    fan_simulation.input['temperature'] = 30
    fan_simulation.compute()
    print("Fan Speed:", fan_simulation.output['fan_speed'])


if __name__ == '__main__':
    print_hi('PyCharm')
