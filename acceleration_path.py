import os 
import imufusion
import numpy as np
import matplotlib.pyplot as plt
from colour import Color
from matplotlib import animation
from dataclasses import dataclass
from scipy.interpolate import interp1d

class AccPath:

    def __init__(self, gyroscope, accelerometer, timestamp, sample_rate, saving_path=""):
        self.gyroscope = gyroscope
        self.accelerometer = accelerometer
        self.timestamp = timestamp
        self.sample_rate = sample_rate
        self.path = saving_path

        self.instantiate_ahrs_algorithm()
        self.process_sensor_data()
        self.identify_moving_periods()
        self.calculate_velocity_with_integral_drift()
        self.find_start_stop_of_moving_period()
        self.remove_integral_drift_from_velocity()
        self.calculate_position()
        self.check_saving_path()

    
    def plot_path(self, title="", n_sections=10, c1="green", c2="blue"):
        section_width = int(len(self.timestamp)/n_sections)
        colors = list(Color(c1).range_to(Color(c2), n_sections))
        plt.figure(figsize=(10, 10))
        ax = plt.axes(projection ='3d')
        x = self.position[:,0]
        y = self.position[:,1]
        z = self.position[:,2]
        for i in range(n_sections):
            i_start = i*section_width
            i_end = i_start+section_width
            ax.plot3D(x[i_start:i_end], y[i_start:i_end], z[i_start:i_end], color=colors[i].hex)
        ax.scatter3D(x[0], y[0], z[0], s=50, color=c1, label="Start")
        ax.scatter3D(x[-1], y[-1], z[-1], s=50, color=c2, label="End")
        if title:
            ax.set_title(title)
        ax.set_xlabel("m")
        ax.set_ylabel("m")
        ax.set_zlabel("m")
        plt.legend()
        plt.savefig(self.path+"path.png")
        plt.show()
        plt.close()
    

    def animate_path(self, length_sec=15, fps=4, c1="green", c2="blue"):
        samples_per_frame = int(len(self.timestamp)/length_sec/fps)
        n_frames = int(length_sec * fps)
        colors = list(Color(c1).range_to(Color(c2), n_frames))
        figure = plt.figure(figsize=(10, 10))
        ax = plt.axes(projection="3d")
        x = self.position[:,0]
        y = self.position[:,1]
        z = self.position[:,2]
        ax.scatter3D(x[0], y[0], z[0], s=50, color=c1, label="Start")
        def update(frame):
            i_start = frame*samples_per_frame
            i_end = i_start+samples_per_frame
            ax.plot3D(x[i_start:i_end], y[i_start:i_end], z[i_start:i_end], color=colors[frame].hex)
            if i_end<6000:
                title=str(round(i_end/100,2))+" s"
            elif i_end<360000:
                title=str(round(i_end/6000,2))+" min"
            else: 
                title=str(round(i_end/360000,2))+" h"
            ax.set_title(title)
            if frame==n_frames-1:
                ax.scatter3D(x[-1], y[-1], z[-1], s=50, color=c2, label="End") 
        anim = animation.FuncAnimation(figure, update,
                                frames=n_frames,
                                interval=1000/fps,
                                repeat=False)
        ax.set_xlabel("m")
        ax.set_ylabel("m")
        ax.set_zlabel("m")
        plt.legend()
        anim.save(self.path+"animation.gif", writer=animation.PillowWriter(fps))
        plt.show()
        plt.close()
    

    def plot_sensor_data(self):
        figure, axes = plt.subplots(nrows=2, sharex=True)
        axes[0].plot(self.timestamp, self.gyroscope[:, 0], "tab:red", label="X")
        axes[0].plot(self.timestamp, self.gyroscope[:, 1], "tab:green", label="Y")
        axes[0].plot(self.timestamp, self.gyroscope[:, 2], "tab:blue", label="Z")
        axes[0].set_title("Gyroscope")
        axes[0].set_ylabel("Degrees/s")
        axes[0].grid()
        axes[1].plot(self.timestamp, self.accelerometer[:, 0], "tab:red", label="X")
        axes[1].plot(self.timestamp, self.accelerometer[:, 1], "tab:green", label="Y")
        axes[1].plot(self.timestamp, self.accelerometer[:, 2], "tab:blue", label="Z")
        axes[1].set_title("Accelerometer")
        axes[1].set_ylabel("g")
        axes[1].grid()
        plt.xlabel("t [s]")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.path+"sensor_data.png")
        plt.show()
        plt.close()


    def plot_euler_angles(self):
        plt.plot(self.timestamp, self.euler[:, 0], "tab:red", label="Roll")
        plt.plot(self.timestamp, self.euler[:, 1], "tab:green", label="Pitch")
        plt.plot(self.timestamp, self.euler[:, 2], "tab:blue", label="Yaw")
        plt.title("Euler angles")
        plt.xlabel("t [s]")
        plt.ylabel("Degrees")
        plt.grid()
        plt.legend()
        plt.savefig(self.path+"euler_angles.png")
        plt.show()
        plt.close()


    def plot_internal_states(self):
        figure, axes = plt.subplots(nrows=3, sharex=True)
        figure.suptitle("AHRS internal states")
        axes[0].plot(self.timestamp, self.internal_states[:, 0])
        axes[0].set_title("Acceleration error")
        axes[0].set_ylabel("Degrees")
        axes[0].grid()
        axes[1].plot(self.timestamp, self.internal_states[:, 1])
        axes[1].set_title("Accelerometer ignored")
        plt.sca(axes[1])
        plt.yticks([0, 1], ["False", "True"])
        axes[1].grid()
        axes[2].plot(self.timestamp, self.internal_states[:, 2])
        axes[2].set_title("Acceleration recovery trigger")
        axes[2].set_xlabel("t [s]")
        axes[2].grid()
        plt.tight_layout()
        plt.savefig(self.path+"internal_states.png")
        plt.show()
        plt.close()


    def plot_acceleration_velocity_position(self):
        figure, axes = plt.subplots(nrows=4, sharex=True)
        # Plot acceleration
        axes[0].plot(self.timestamp, self.acceleration[:, 0], "tab:red", label="X")
        axes[0].plot(self.timestamp, self.acceleration[:, 1], "tab:green", label="Y")
        axes[0].plot(self.timestamp, self.acceleration[:, 2], "tab:blue", label="Z")
        axes[0].set_title("Acceleration")
        axes[0].set_ylabel("m/s/s")
        axes[0].grid()
        axes[0].legend()
        # Plot moving periods
        axes[1].plot(self.timestamp, self.is_moving, "tab:cyan")
        axes[1].set_title("Is moving")
        plt.sca(axes[1])
        plt.yticks([0, 1], ["False", "True"])
        axes[1].grid()
        # Plot velocity
        axes[2].plot(self.timestamp, self.velocity[:, 0], "tab:red", label="X")
        axes[2].plot(self.timestamp, self.velocity[:, 1], "tab:green", label="Y")
        axes[2].plot(self.timestamp, self.velocity[:, 2], "tab:blue", label="Z")
        axes[2].set_title("Velocity")
        axes[2].set_ylabel("m/s")
        axes[2].grid()
        # Plot position
        axes[3].plot(self.timestamp, self.position[:, 0], "tab:red", label="X")
        axes[3].plot(self.timestamp, self.position[:, 1], "tab:green", label="Y")
        axes[3].plot(self.timestamp, self.position[:, 2], "tab:blue", label="Z")
        axes[3].set_title("Position")
        axes[3].set_xlabel("Seconds")
        axes[3].set_ylabel("m")
        axes[3].grid()
        plt.tight_layout()
        plt.savefig(self.path+"acceleration_velocity_position.png")
        plt.show()
        plt.close()
    

    def print_distance_start_final(self):
        # Print distance between start and final positions
        print("Start-Final-Distance: " + "{:.3f}".format(np.sqrt(self.position[-1].dot(self.position[-1]))) + " m")


    def instantiate_ahrs_algorithm(self):
        self.offset = imufusion.Offset(self.sample_rate)
        self.ahrs = imufusion.Ahrs()
        self.ahrs.settings = imufusion.Settings(
            imufusion.CONVENTION_NWU,
            0.5,  # gain
            2000,  # gyroscope range
            10,  # acceleration rejection
            0,  # magnetic rejection
            5 * self.sample_rate)  # rejection timeout = 5 seconds


    def process_sensor_data(self):
        self.delta_time = np.diff(self.timestamp, prepend=self.timestamp[0])
        self.euler = np.empty((len(self.timestamp), 3))
        self.internal_states = np.empty((len(self.timestamp), 3))
        self.acceleration = np.empty((len(self.timestamp), 3))

        for index in range(len(self.timestamp)):
            self.gyroscope[index] = self.offset.update(self.gyroscope[index])

            self.ahrs.update_no_magnetometer(
                self.gyroscope[index], 
                self.accelerometer[index], 
                self.delta_time[index])

            self.euler[index] = self.ahrs.quaternion.to_euler()

            ahrs_internal_states = self.ahrs.internal_states
            self.internal_states[index] = np.array([
                ahrs_internal_states.acceleration_error,
                ahrs_internal_states.accelerometer_ignored,
                ahrs_internal_states.acceleration_recovery_trigger])

            self.acceleration[index] = 9.81 * self.ahrs.earth_acceleration  # convert g to m/s/s


    def identify_moving_periods(self):
        self.is_moving = np.empty(len(self.timestamp))

        for index in range(len(self.timestamp)):
            self.is_moving[index] = np.sqrt(
                self.acceleration[index].dot(self.acceleration[index])) > 3  # threshold = 3 m/s/s

        margin = int(0.1 * self.sample_rate)  # 100 ms

        for index in range(len(self.timestamp) - margin):
            self.is_moving[index] = any(self.is_moving[index:(index + margin)])  # add leading margin

        for index in range(len(self.timestamp) - 1, margin, -1):
            self.is_moving[index] = any(self.is_moving[(index - margin):index])  # add trailing margin


    def calculate_velocity_with_integral_drift(self):
        # Calculated velocity includes integral drift
        self.velocity = np.zeros((len(self.timestamp), 3))

        for index in range(len(self.timestamp)):
            if self.is_moving[index]:  # only integrate if moving
                self.velocity[index] = self.velocity[index - 1] + self.delta_time[index] * self.acceleration[index]


    def find_start_stop_of_moving_period(self):
        # Find start and stop indices of each moving period
        is_moving_diff = np.diff(self.is_moving, append=self.is_moving[-1])

        @dataclass
        class IsMovingPeriod:
            start_index: int = -1
            stop_index: int = -1

        self.is_moving_periods = []
        is_moving_period = IsMovingPeriod()

        for index in range(len(self.timestamp)):
            if is_moving_period.start_index == -1:
                if is_moving_diff[index] == 1:
                    is_moving_period.start_index = index

            elif is_moving_period.stop_index == -1:
                if is_moving_diff[index] == -1:
                    is_moving_period.stop_index = index
                    self.is_moving_periods.append(is_moving_period)
                    is_moving_period = IsMovingPeriod()


    def remove_integral_drift_from_velocity(self):
        # Remove integral drift from velocity
        velocity_drift = np.zeros((len(self.timestamp), 3))

        for is_moving_period in self.is_moving_periods:
            start_index = is_moving_period.start_index
            stop_index = is_moving_period.stop_index

            t = [self.timestamp[start_index], self.timestamp[stop_index]]
            x = [self.velocity[start_index, 0], self.velocity[stop_index, 0]]
            y = [self.velocity[start_index, 1], self.velocity[stop_index, 1]]
            z = [self.velocity[start_index, 2], self.velocity[stop_index, 2]]

            t_new = self.timestamp[start_index:(stop_index + 1)]

            velocity_drift[start_index:(stop_index + 1), 0] = interp1d(t, x)(t_new)
            velocity_drift[start_index:(stop_index + 1), 1] = interp1d(t, y)(t_new)
            velocity_drift[start_index:(stop_index + 1), 2] = interp1d(t, z)(t_new)

        self.velocity = self.velocity - velocity_drift


    def calculate_position(self):
        self.position = np.zeros((len(self.timestamp), 3))
        for index in range(len(self.timestamp)):
            self.position[index] = self.position[index - 1] + self.delta_time[index] * self.velocity[index]


    def check_saving_path(self):
        if self.path:
            if not os.path.exists(self.path): 
                os.makedirs(self.path)
