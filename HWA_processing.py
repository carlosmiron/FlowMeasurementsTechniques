import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


############################################## Functions #################################################
def txt_to_df(filename):
    # Open the file
    with open(filename, 'r') as file:
        # Read the file into a dataframe where each tab separates the columns and each "\n" separates the rows
        lines = file.readlines()
        lines = lines[23:]      #Make it go to -1 if you want to remove the None at the end
        #Use the tab separater to split into two columns
        lines = [line.split('\t') for line in lines]
        # Remove the "\n" characters
        lines = [[element.strip() for element in line] for line in lines]

    #Turn it into a datafram with headers "X_Value" and "Voltage"
    df = pd.DataFrame(lines, columns = ["X_Value", "Voltage"])
    
    return df

#Data is currently processed by taking the average of all measured voltages in a file
def avg_voltage(df):
    # Calculate the average voltage
    avg_voltage = df['Voltage'].astype(float).mean()
    return avg_voltage

def voltage_to_velocity(coefficients, voltage):
    return np.polyval(coefficients, voltage)


############################################## Calibration #################################################
voltages = np.zeros(11)
velocities = np.zeros(11)
for i in range(0, 21, 2):
    if i < 10:
        filename = 'HWA/Calibration_00' + str(i) + '.txt'
    else:
        filename = 'HWA/Calibration_0' + str(i) + '.txt'

    voltages[i//2] = avg_voltage(txt_to_df(filename))
    velocities[i//2] = i



polyfit_coefficients = np.polyfit(voltages, velocities, 4)




#Plot the calibration data (with voltages on the y axis, but we can also place them on the x axis)
calibration_plot = True
if calibration_plot:
    voltages_plot = np.linspace(min(voltages), max(voltages), 100)

    fig = plt.figure()
    plt.plot(np.polyval(polyfit_coefficients, voltages_plot), voltages_plot, color = "red", zorder = 1, label = "Polynomial fit")
    plt.scatter(velocities, voltages, color = "black", zorder = 2, label = "Calibration points")

    plt.legend()
    plt.xlabel('Velocity')
    plt.ylabel('Voltage')
    plt.title('Voltage vs Velocity')
    plt.show()

############################################## Velocity Analysis #################################################
angles = ["00", "05", "15"]
positions = [str(i) for i in range(-40, 44, 4)]

for i in range(len(positions)):
    if len(positions[i]) == 1:
        positions[i] = "0" + positions[i]
    if len(positions[i]) == 2 and positions[i][0] == "-":
        positions[i] = "-0" + positions[i][1]
    if int(positions[i]) >= 0:
        positions[i] = "+" + positions[i]


mean_velocities_00 = np.zeros(len(positions))
mean_velocities_05 = np.zeros(len(positions))
mean_velocities_15 = np.zeros(len(positions))
fluc_velocities_00 = np.zeros(len(positions))
fluc_velocities_05 = np.zeros(len(positions))
fluc_velocities_15 = np.zeros(len(positions))

for angle in angles:
    for position in positions:
        filename = 'HWA/Mesruements_' + angle + '_' + position + '.txt'
        df = txt_to_df(filename)
        mean_voltage = avg_voltage(df)
        mean_vel = np.polyval(polyfit_coefficients, mean_voltage)

        voltages_array = np.array(df['Voltage'].astype(float))
        #Remove all instances of nan
        voltages_array = voltages_array[~np.isnan(voltages_array)]
        
        velocities_array = np.polyval(polyfit_coefficients, voltages_array)
        fluctuations_array = velocities_array - mean_vel
        fluctuations_squared_array = fluctuations_array**2
        fluc_sqrd_mean = np.mean(fluctuations_squared_array)
        fluc_sqrd_mean = np.sqrt(fluc_sqrd_mean)


        if angle == "00":
            mean_velocities_00[positions.index(position)] = mean_vel
            fluc_velocities_00[positions.index(position)] = fluc_sqrd_mean
        elif angle == "05":
            mean_velocities_05[positions.index(position)] = mean_vel
            fluc_velocities_05[positions.index(position)] = fluc_sqrd_mean
        elif angle == "15":
            mean_velocities_15[positions.index(position)] = mean_vel
            fluc_velocities_15[positions.index(position)] = fluc_sqrd_mean
        


#Plot the mean velocities
mean_velocity_plot = False
if mean_velocity_plot:
    fig = plt.figure()
    
    plt.plot(mean_velocities_00, positions, label = r'$\alpha = 0$', marker = "x")
    plt.plot(mean_velocities_05, positions, label = r'$\alpha = 5$', marker = "o", mfc = "none")
    plt.plot(mean_velocities_15, positions, label = r'$\alpha = 15$', marker = "v", mfc = "none")
    plt.legend()
    plt.xlabel('Velocity [m/s]')
    plt.ylabel('Position [mm]')
    plt.title('Mean Velocity vs Position')
    plt.show()



#Plot the fluctuations
fluc_velocity_plot = False
if fluc_velocity_plot:
    fig = plt.figure()
    
    plt.plot(fluc_velocities_00, positions, label = r'$\alpha = 0$', marker = "x")
    plt.plot(fluc_velocities_05, positions, label = r'$\alpha = 5$', marker = "o", mfc = "none")
    plt.plot(fluc_velocities_15, positions, label = r'$\alpha = 15$', marker = "v", mfc = "none")
    plt.legend()
    plt.xlabel('Fluctuating Velocity [m/s]')
    plt.ylabel('Position [mm]')
    plt.title('Fluctuation vs Position')
    plt.show()

############################################## Spectral Analysis #################################################

#Compute the fourier tranform of a sine wave using scipy
from scipy.fft import rfft, rfftfreq

#Create a sine wave
fs = 1000       #Sampling rate. #Average number of samples obtained in one second. Inverse of sampling period
f = 10
t = np.linspace(0, 1, fs)
sine_wave = np.sin(2*np.pi*f*t)

#Compute the fourier transform
sine_wave_fft = rfft(sine_wave)
freqs = rfftfreq(len(sine_wave), 1/fs)

#Plot the fourier transform
sine_wave_plot = True
if sine_wave_plot:
    fig = plt.figure()
    plt.plot(freqs, np.abs(sine_wave_fft))
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')
    plt.title('Fourier Transform of a Sine Wave')
    plt.show()



"""	
#Print a random df
voltage_example = txt_to_df('HWA/Mesruements_00_+40.txt')
print(voltage_example)
velocities_example = voltage_to_velocity(polyfit_coefficients, voltage_example['Voltage'].astype(float))
print(velocities_example)

#Remove nans from the array
velocities_example = velocities_example[~np.isnan(velocities_example)]
print(velocities_example)

#Plot the velocity signal

fig = plt.figure()
plt.plot(velocities_example)
plt.xlabel('Time')
plt.ylabel('Velocity')
plt.title('Velocity Signal')
plt.show()
"""

