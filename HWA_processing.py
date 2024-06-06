import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import rfft, rfftfreq



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

def fourier(signal, Ts):
    #Compute the fourier transform
    signal_fft = rfft(signal)
    freqs = rfftfreq(len(signal), Ts)
    return signal_fft, freqs


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
calibration_plot = False
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
        fluc_sqrd_mean = np.std(velocities_array, ddof=1)


        if angle == "00":
            mean_velocities_00[positions.index(position)] = mean_vel
            fluc_velocities_00[positions.index(position)] = fluc_sqrd_mean
        elif angle == "05":
            mean_velocities_05[positions.index(position)] = mean_vel
            fluc_velocities_05[positions.index(position)] = fluc_sqrd_mean
        elif angle == "15":
            mean_velocities_15[positions.index(position)] = mean_vel
            fluc_velocities_15[positions.index(position)] = fluc_sqrd_mean

#sampling time and turbulence intensity from correlation file
Cor_data = np.genfromtxt('HWA/CorrelationTest.txt', delimiter='\t', skip_header=23, names=['Time', 'Voltage'], dtype=None, encoding=None)
Cor_time = Cor_data['Time']
Cor_voltage = Cor_data['Voltage']
Cor_velocity = np.polyval(polyfit_coefficients, Cor_voltage)
# Mean and standard deviation
mu_Cor = np.mean(Cor_velocity)
std_Cor = np.std(Cor_velocity, ddof=1)
u_Cor = Cor_velocity - mu_Cor

# Turbulence intensity
Tu = std_Cor / mu_Cor

#uncertainty less than 1% with confidence level 99.7%
epsilon = 0.01
k = 3

# Calculate autocorrelation rho(tau)
full_corr = np.correlate(u_Cor, u_Cor, mode='full') / np.mean((u_Cor)**2 * len(u_Cor))
rho_tau = full_corr[len(u_Cor) - 1:]
#find first zero point in rho(tau)
sign_changes = np.diff(np.sign(rho_tau))
zero_crossings = np.where(sign_changes)[0]  # indices where the sign change occurs
T1 = Cor_time[zero_crossings[0]] + rho_tau[zero_crossings[0]] / abs(rho_tau[zero_crossings[0]+1] - rho_tau[zero_crossings[0]]) * abs(Cor_time[zero_crossings[0]+1] - Cor_time[zero_crossings[0]])
T_sample = 2 * T1 * std_Cor**2 * (k / (mu_Cor * epsilon))**2
f_sample = 1 / (2 * T1)

# Plotting for sample time
sample_time_plot = False
if sample_time_plot:
    print('Minimum sample frequency: ', f_sample, 'Hz')
    print('Sample Time: ', T_sample, 's')
    plt.figure(figsize=(10, 5))
    plt.plot(Cor_time*1000, rho_tau, label='Normalized Autocorrelation')
    plt.axhline(0, color='red', linestyle='--', label='Zero Crossing')
    plt.title('Normalized Autocorrelation Function of Velocity Fluctuations')
    plt.xlabel('time [ms]')
    plt.ylabel(r'$\rho$')
    plt.xlim([0, 100])
    plt.legend()
    plt.show()



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


#Initialize fft_dict as dictionary 
fft_dict = {}
freqs_dict = {}
#Give it keys for each position and angle
for angle in angles:
    fft_dict[angle] = {}
    freqs_dict[angle] = {}
    for position in positions:
        fft_dict[angle][position] = []
        freqs_dict[angle][position] = []



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

        #Compute the fourier transform
        fluctuations_fft, freqs = fourier(fluctuations_array, T_sample) 

        if angle == "00":
            fft_dict[angle][position].append(np.abs(fluctuations_fft))
            freqs_dict[angle][position].append(freqs)
        elif angle == "05":
            fft_dict[angle][position].append(np.abs(fluctuations_fft))
            freqs_dict[angle][position].append(freqs)
        elif angle == "15":
            fft_dict[angle][position].append(np.abs(fluctuations_fft))
            freqs_dict[angle][position].append(freqs)



fourier_plot_00 = True
if fourier_plot_00:     
    fig = plt.figure()
    #Plot the fourier transform of angle is 0 and every four positions
    for i in range(0, len(positions), 4):
        plt.plot(freqs_dict["00"][positions[i]][0], fft_dict["00"][positions[i]][0], label = positions[i])

    plt.legend()
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')
    plt.title('Fourier Transform of the Fluctuations Signal')
    plt.show()


fourier_plot_05 = False
if fourier_plot_05:     
    fig = plt.figure()
    #Plot the fourier transform of angle is 0 and every four positions
    for i in range(0, len(positions), 4):
        plt.plot(freqs_dict["05"][positions[i]][0], fft_dict["05"][positions[i]][0], label = positions[i])

    plt.legend()
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')
    plt.title('Fourier Transform of the Fluctuations Signal')
    plt.show()
    

fourier_plot_15 = False
if fourier_plot_15:     
    fig = plt.figure()
    #Plot the fourier transform of angle is 0 and every four positions
    for i in range(0, len(positions), 4):
        plt.plot(freqs_dict["15"][positions[i]][0], fft_dict["15"][positions[i]][0], label = positions[i])

    plt.legend()
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')
    plt.title('Fourier Transform of the Fluctuations Signal')
    plt.show()

