import numpy as np

def optimizeSignal(signal, vehicles):
    numSignals = 3
    defaultTimer = 60
    shortTimer = 10
    vehicleLimit = 20
    cycles = 50


    trafficData = {signal: vehicles}
    signalTimers = {signal: defaultTimer}

    def checkTraffic(timer, cars):
        return 1 if cars < vehicleLimit else 1 / (1 + (cars - vehicleLimit))

    for i in range(cycles):
        fitness = {sig: checkTraffic(signalTimers[sig], trafficData[sig]) for sig in signalTimers}
        chances = np.array(list(fitness.values())) / sum(fitness.values())
        chosenSignals = np.random.choice(list(signalTimers.keys()), numSignals, p=chances)

        for sig in chosenSignals:
            if trafficData[sig] >= vehicleLimit:
                signalTimers[sig] = shortTimer

    print(f"Optimized Timer for {signal}: {signalTimers[signal]} sec")

3
