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
mean_velocity_plot = True
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
fluc_velocity_plot = True
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




