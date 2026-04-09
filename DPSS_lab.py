import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

data_1064_current = [
    90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230,
    240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 370, 380, 390, 400, 
]

data_1064_mv = [
    -1.3, 0.4, 0.4, 1.3, 2.1, 3.0, 3.8, 4.9, 6.0, 6.8, 7.0, 8.9,
    9.8, 11.1, 12.0, 13.4, 14.7, 15.4, 16.0, 17.3, 19.3,
    21.4, 22.4, 23.6, 25.6, 28.6, 30.2, 32.5, 34.0, 37.4, 41.5,
]

data_808_current = [
    90, 100, 110, 120, 130, 140, 150, 160, 170, 180,
    190, 200, 210, 220, 230, 240, 250, 260, 270, 280,
    290, 300, 310, 320, 330, 340, 350, 360, 370, 380,
    390, 400, #410, 420, 430, 440, 450, 460
]

data_808_mv = [
    19.5, 36.5, 52.42, 68, 87.6, 102.5, 120.5, 136.5, 150, 166,
    170, 186, 199, 215, 229, 244, 256, 271, 287, 298,
    311, 325, 339, 354, 365, 378, 393, 404, 420, 436,
    444, #447, #448, 449, 449, 450, 450, 450
]

### Impedance used to measure the power
imp = 50

### convert data to power

data_1064_power = [((x/1000)**2)/imp for x in data_1064_mv]
data_808_power = [((x/1000)**2)/imp for x in data_808_mv]

### Fit a exponetial to our data

def exp_func(x, a, b, c):
    return a * np.exp(b * x) + c

def linear(x, m, b):
    return m*x+b

popt, pcov = curve_fit(exp_func, data_808_power, data_1064_power, p0=(max(data_1064_power), 10, 0))
#linpopt, linpcov = curve_fit(linear, data_808_power, data_808_power, p0=(1,0))
a, b, c = popt

print("Data Slope ", popt[1])

#y_fit_lin = [linear(x, linpopt[0], linpopt[1]) for x in data_808_power]
y_fit = [exp_func(x, a, b, c) for x in data_808_power]

# choose indices for linear region
x_lin = data_808_power[3:]   # example — YOU should adjust this
y_lin = data_1064_power[3:]

coeffs = np.polyfit(x_lin, y_lin, 1)
slope_eff = coeffs[0]
intercept = coeffs[1]

print("Slope_eff", slope_eff)
G_double = 1 / (1 - slope_eff)
print("G_double", G_double)


#print(y_fit_lin)
plt.title("Pump Power vs Crystal Output Power")
plt.plot(data_808_power, y_fit, label="Output Power", linestyle="--")
#plt.plot(data_808_power, y_fit_lin, label="Input Fit", linestyle="--")
#plt.plot(data_1064_current,data_1064_mv, "b*")
plt.plot(data_808_power, data_1064_power, "b*")
plt.plot(data_808_power,data_808_power, "r", label="Input Power")
plt.xlabel("Power (w)")
plt.ylabel("Power (w)")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.grid()
plt.show()