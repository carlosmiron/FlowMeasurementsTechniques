import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import welch
from scipy.signal import find_peaks


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

#Data is currently processed by taking the average of all measured voltages in a file without outliers for the polynomail fit
def avg_voltage_poly(df):
    # Calculate the average voltage
    avg_voltage = df['Voltage'].astype(float).mean()
    std_voltage = df['Voltage'].astype(float).std(ddof=1)
    voltage =  df['Voltage'].astype(float)
    valid_indices = (voltage < (avg_voltage + 3 * std_voltage)) & (voltage > (avg_voltage - 3 * std_voltage))
    voltage = voltage[valid_indices]
    avg_voltage = np.mean(voltage)
    return avg_voltage

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

def fourier_with_windowing(signal, Ts, window):
    fs = 1 / Ts
    # Compute the power spectral density using Welch's method
    freqs, psd = welch(signal, fs=fs, nperseg=window)
    return psd, freqs


############################################## Calibration #################################################
voltages = np.zeros(11)
velocities = np.zeros(11)
for i in range(0, 21, 2):
    if i < 10:
        filename = 'HWA/Calibration_00' + str(i) + '.txt'
    else:
        filename = 'HWA/Calibration_0' + str(i) + '.txt'

    voltages[i//2] = avg_voltage_poly(txt_to_df(filename))
    velocities[i//2] = i



polyfit_coefficients = np.polyfit(voltages, velocities, 4)

outlier_plot = False
if outlier_plot:
    Outlier_data = np.genfromtxt('HWA/Calibration_020.txt', delimiter='\t', skip_header=23, names=['Time', 'Voltage'],
                             dtype=None, encoding=None)
    Outlier_time = Outlier_data['Time']
    Outlier_voltage = Outlier_data['Voltage']
    Outlier_velocity = np.polyval(polyfit_coefficients, Outlier_voltage)
    # Mean and standard deviation
    mu_Outlier = np.mean(Outlier_velocity)
    std_Outlier = np.std(Outlier_velocity, ddof=1)
    u_Outlier = Outlier_velocity - mu_Outlier
    u_Outlier_mean = np.mean(u_Outlier)
    u_Outlier_std = np.std(u_Outlier, ddof=1)
    fig = plt.figure()

    plt.plot(Outlier_time, u_Outlier)
    plt.axhline(u_Outlier_mean + 3 * u_Outlier_std, color='red', linestyle='--', label='$\mu$ +/- 3 * $\sigma$')
    plt.axhline(u_Outlier_mean - 3 * u_Outlier_std, color='red', linestyle='--')
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('Fluctuation [m/s]')
    plt.grid()
    plt.show()




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
    plt.grid()
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
u_Cor_mean = np.mean(u_Cor)
u_Cor_std = np.std(u_Cor, ddof=1)
u_rms = np.sqrt(np.mean(np.square(u_Cor)))

'''#detect outliers and remove from data not done for sampling time because of time plotting
valid_indices = (Cor_velocity < (mu_Cor + 3 * std_Cor)) & (Cor_velocity > (mu_Cor - 3 * std_Cor))
print(valid_indices)
print(len(valid_indices))
Cor_velocity_check = Cor_velocity[valid_indices]
Cor_velocity = Cor_velocity[Cor_velocity < (mu_Cor + 3*std_Cor)]
Cor_velocity = Cor_velocity[Cor_velocity > (mu_Cor - 3*std_Cor)]
Cor_time = Cor_time[valid_indices]
u_Cor = Cor_velocity - mu_Cor'''

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
T_sample = 2 * T1 * u_rms**2 * (k / (mu_Cor * epsilon))**2
f_sample = 2 / T1

# Plotting for sample time
sample_time_plot = False
if sample_time_plot:
    print('Minimum number of uncorrelated samples: ', u_rms**2 * (k / (mu_Cor * epsilon))**2)
    print('Minimum sample frequency: ', f_sample, 'Hz')
    print('Sample Time: ', T_sample, 's')
    plt.figure(figsize=(10, 5))
    plt.plot(Cor_time*1000, rho_tau, label='Normalized Autocorrelation', zorder=2)
    plt.axhline(0, color='red', linestyle='--', label='Zero Crossing', zorder=1)
    plt.scatter(T1*1000, 0, s=50, marker='o', color='black', zorder=3, label=f'$T_I$={np.round(T1*1000, 3)}ms')
    plt.xlabel('time [ms]')
    plt.ylabel(r'$\rho$')
    plt.xlim([0, 100])
    plt.legend()
    plt.grid()
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
        #romove outliers
        '''mu_array = np.mean(velocities_array)
        std_array = np.std(velocities_array, ddof=1)
        valid_indices = (velocities_array < (mu_array + 3 * std_array)) & (velocities_array > (mu_array - 3 * std_array))
        velocities_array = velocities_array[valid_indices]
        fluctuations_array = velocities_array - mean_vel'''

        #Compute the fourier transform
        #fluctuations_fft, freqs = fourier(fluctuations_array, 9.765625E-5)
        fluctuations_fft, freqs = fourier_with_windowing(fluctuations_array, 9.765625E-5, 2000)

        if angle == "00":
            fft_dict[angle][position].append(np.abs(fluctuations_fft))
            freqs_dict[angle][position].append(freqs)
        elif angle == "05":
            fft_dict[angle][position].append(np.abs(fluctuations_fft))
            freqs_dict[angle][position].append(freqs)
        elif angle == "15":
            fft_dict[angle][position].append(np.abs(fluctuations_fft))
            freqs_dict[angle][position].append(freqs)


fourier_plot_00 = False
if fourier_plot_00:
    # Create a figure and a set of subplots
    fig, axs = plt.subplots(3, 2, figsize=(15, 10))  # figsize can be adjusted based on your display preferences

    # Flatten the axs array for easier indexing
    axs = axs.flatten()

    # Track subplot index
    index = 0

    # Iterate through your positions, plotting every fourth position as specified
    for i in [7, 8, 9, 10, 11, 12]:
        ax = axs[index]
        frequencies = freqs_dict["00"][positions[i]][0]
        amplitudes = fft_dict["00"][positions[i]][0]

        # Plot the Fourier Transform
        ax.loglog(frequencies, amplitudes, linewidth=1, label='$y_{pos}$=' + f'{positions[i]}cm')

        # Calculate a dynamic prominence threshold based on the standard deviation of amplitudes
        prominence_threshold = np.std(amplitudes) * 3  # Modify the factor as needed to tune sensitivity

        # Find peaks with dynamic prominence
        peaks, properties = find_peaks(amplitudes, prominence=prominence_threshold)
        peak_freqs = frequencies[peaks]
        peak_amps = amplitudes[peaks]
        valid_indices = (peak_freqs > 20)
        peak_freqs = peak_freqs[valid_indices]
        peak_amps = peak_amps[valid_indices]

        ax.scatter(peak_freqs, peak_amps, color='red', s=25, zorder=5, label='peaks')
        last_annotated_freq = None  # Track the height of the last annotation
        offset = 2
        for idx, (freq, amp) in enumerate(zip(peak_freqs, peak_amps)):
            # Dynamic vertical offset
            if last_annotated_freq and abs(last_annotated_freq - freq) < last_annotated_freq and offset != 10:
                offset = 10  # Move text down if the previous annotation was close in amplitude
            else:
                offset = 2  # Default upward offset

            last_annotated_freq = freq  # Update the last annotated height

            ax.annotate(f'{freq:.1f} Hz', xy=(freq, amp), xytext=(0, offset),
                        textcoords="offset points", ha='center', va='bottom')

        # Set titles, labels, etc.
        ax.set_xlabel('f [Hz]')
        ax.set_ylabel('PSD [$m^2/s^2/Hz$]')
        ax.set_xlim([10, 5000])
        ax.set_ylim([np.min(amplitudes), 5 * np.max(amplitudes)])
        ax.legend()

        # Increment the subplot index
        index += 1

        # Break if all subplots are filled
        if index >= 6:
            break

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()


fourier_plot_05 = True
if fourier_plot_05:     
    # Create a figure and a set of subplots
    fig, axs = plt.subplots(3, 2, figsize=(15, 10))  # figsize can be adjusted based on your display preferences

    # Flatten the axs array for easier indexing
    axs = axs.flatten()

    # Track subplot index
    index = 0

    # Iterate through your positions, plotting every fourth position as specified
    for i in [5, 6, 7, 8, 9, 10]:
        ax = axs[index]
        frequencies = freqs_dict["05"][positions[i]][0]
        amplitudes = fft_dict["05"][positions[i]][0]

        # Plot the Fourier Transform
        ax.loglog(frequencies, amplitudes, linewidth=1, label='$y_{pos}$=' + f'{positions[i]}cm')

        # Calculate a dynamic prominence threshold based on the standard deviation of amplitudes
        prominence_threshold = np.std(amplitudes) * 3  # Modify the factor as needed to tune sensitivity

        # Find peaks with dynamic prominence
        peaks, properties = find_peaks(amplitudes, prominence=prominence_threshold)
        peak_freqs = frequencies[peaks]
        peak_amps = amplitudes[peaks]
        valid_indices = (peak_freqs > 20)
        peak_freqs = peak_freqs[valid_indices]
        peak_amps = peak_amps[valid_indices]
        ax.scatter(peak_freqs, peak_amps, color='red', s=25, zorder=5, label='peaks')
        last_annotated_freq = None  # Track the height of the last annotation
        last_annotated_amp = 0
        offset = 2
        for idx, (freq, amp) in enumerate(zip(peak_freqs, peak_amps)):
            # Dynamic vertical offset
            if last_annotated_freq and 2*last_annotated_freq > freq and offset != 18:
                offset = 10
                if last_annotated_amp > amp:
                    offset = 18  # Move text down if the previous annotation was close in amplitude
            else:
                offset = 2  # Default upward offset

            if np.round(freq, 1) == 204.8 or np.round(freq, 1) == 276.5 or np.round(freq, 1) == 312.3 or amp == np.max(peak_amps):
                ax.annotate(f'{freq:.1f} Hz', xy=(freq, amp), xytext=(0, offset),
                        textcoords="offset points", ha='center', va='bottom')
                last_annotated_freq = freq  # Update the last annotated height
                last_annotated_amp = amp

        # Set titles, labels, etc.
        ax.set_xlabel('f [Hz]')
        ax.set_ylabel('PSD [$m^2/s^2/Hz$]')
        ax.set_xlim([10, 5000])
        ax.set_ylim([np.min(amplitudes), 5 * np.max(amplitudes)])
        ax.legend()

        # Increment the subplot index
        index += 1

        # Break if all subplots are filled
        if index >= 6:
            break

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()
    

fourier_plot_15 = False
if fourier_plot_15: 
    # Create a figure and a set of subplots
    fig, axs = plt.subplots(3, 2, figsize=(15, 10))  # figsize can be adjusted based on your display preferences

    # Flatten the axs array for easier indexing
    axs = axs.flatten()

    # Track subplot index
    index = 0

    # Iterate through your positions, plotting every fourth position as specified
    for i in [4, 5, 6, 12, 13, 14]:
    #for i in range(7,13,1):
        # Select the next subplot
        ax = axs[index]
        frequencies = freqs_dict["15"][positions[i]][0]
        amplitudes = fft_dict["15"][positions[i]][0]

        # Plot the Fourier Transform
        ax.loglog(frequencies, amplitudes, linewidth=1, label='$y_{pos}$=' + f'{positions[i]}cm')

        # Calculate a dynamic prominence threshold based on the standard deviation of amplitudes
        prominence_threshold = np.std(amplitudes) * 2  # Modify the factor as needed to tune sensitivity

        # Find peaks with dynamic prominence
        peaks, properties = find_peaks(amplitudes, prominence=prominence_threshold)
        peak_freqs = frequencies[peaks]
        peak_amps = amplitudes[peaks]
        valid_indices = (peak_freqs > 20)
        peak_freqs = peak_freqs[valid_indices]
        peak_amps = peak_amps[valid_indices]
        # Annotate peaks with their frequencies
        '''ax.scatter(peak_freqs, peak_amps, color='red', s=25, zorder=5, label='peaks')
        for freq, amp in zip(peak_freqs, peak_amps):
            ax.annotate(f'{freq:.001f} Hz', xy=(freq, amp), xytext=(0, 2),
                        textcoords="offset points", ha='center', va='bottom')'''

        ax.scatter(peak_freqs, peak_amps, color='red', s=25, zorder=5, label='peaks')
        last_annotated_freq = None  # Track the height of the last annotation
        offset = 2
        for idx, (freq, amp) in enumerate(zip(peak_freqs, peak_amps)):
            # Dynamic vertical offset
            if last_annotated_freq and abs(last_annotated_freq - freq) < last_annotated_freq and offset != 10:
                offset = 10  # Move text down if the previous annotation was close in amplitude
            else:
                offset = 2  # Default upward offset

            last_annotated_freq = freq  # Update the last annotated height

            ax.annotate(f'{freq:.1f} Hz', xy=(freq, amp), xytext=(0, offset),
                        textcoords="offset points", ha='center', va='bottom')

        # Set titles, labels, etc.
        ax.set_xlabel('f [Hz]')
        ax.set_ylabel('PSD [$m^2/s^2/Hz$]')
        ax.set_xlim([10, 5000])
        ax.set_ylim([np.min(amplitudes), 5*np.max(amplitudes)])
        ax.legend()

        # Increment the subplot index
        index += 1

        # Break if all subplots are filled
        if index >= 6:
            break

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()

    

