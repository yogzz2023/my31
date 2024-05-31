import numpy as np
import math
import csv
import pandas as pd
from scipy.stats import chi2
import matplotlib.pyplot as plt

r = []
el = []
az = []

class CVFilter:
    def __init__(self):
        # Initialize filter parameters
        self.Sf = np.zeros((6, 1))  # Filter state vector
        self.Pf = np.eye(6)  # Filter state covariance matrix
        self.Sp = np.zeros((6, 1))
        self.plant_noise = 20  # Plant noise covariance
        self.H = np.eye(3, 6)  # Measurement matrix
        self.R = np.eye(3)  # Measurement noise covariance
        self.S = np.eye(3)  # Measurement noise covariance with specific values
        self.S[0, 0] = 0.1
        self.S[1, 1] = 0.1
        self.S[2, 2] = 0.1
        self.Meas_Time = 0  # Measured time
        self.Z = np.zeros((3, 1))

    def initialize_filter_state(self, x, y, z, vx, vy, vz, time):
        # Initialize filter state
        self.Sf = np.array([[x], [y], [z], [vx], [vy], [vz]])
        self.Meas_Time = time

    def initialize_measurement_for_filtering(self, x, y, z, mt):
        self.Z = np.array([[x], [y], [z]])
        self.Meas_Time = mt

    def predict_step(self, current_time):
        # Predict step
        dt = current_time - self.Meas_Time
        Phi = np.eye(6)
        Phi[0, 3] = dt
        Phi[1, 4] = dt
        Phi[2, 5] = dt
        Q = np.eye(6) * self.plant_noise
        self.Sp = np.dot(Phi, self.Sf)
        self.Pf = np.dot(np.dot(Phi, self.Pf), Phi.T) + Q

    def update_step(self):
        # Update step
        Inn = self.Z - np.dot(self.H, self.Sf)
        S = np.dot(self.H, np.dot(self.Pf, self.H.T)) + self.R
        K = np.dot(np.dot(self.Pf, self.H.T), np.linalg.inv(S))
        self.Sf = self.Sf + np.dot(K, Inn)
        self.Pf = np.dot(np.eye(6) - np.dot(K, self.H), self.Pf)

def sph2cart(az, el, r):
    x = r * np.cos(el * np.pi / 180) * np.sin(az * np.pi / 180)
    y = r * np.cos(el * np.pi / 180) * np.cos(az * np.pi / 180)
    z = r * np.sin(el * np.pi / 180)
    return x, y, z

def cart2sph(x, y, z):
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    el = math.atan(z / np.sqrt(x ** 2 + y ** 2)) * 180 / np.pi
    az = math.atan(y / x)

    if x > 0.0:
        az = 3.14 / 2 - az
    else:
        az = 3 * 3.14 / 2 - az

    az = az * 180 / np.pi

    if az < 0.0:
        az = 360 + az

    if az > 360:
        az = az - 360

    return r, az, el

def cart2sph2(x, y, z):
    r, az, el = [], [], []
    for i in range(len(x)):
        r_val = np.sqrt(x[i]**2 + y[i]**2 + z[i]**2)
        el_val = math.atan(z[i] / np.sqrt(x[i]**2 + y[i]**2)) * 180 / np.pi
        az_val = math.atan(y[i] / x[i])
        
        if x[i] > 0.0:
            az_val = 3.14 / 2 - az_val
        else:
            az_val = 3 * 3.14 / 2 - az_val

        az_val = az_val * 180 / np.pi

        if az_val < 0.0:
            az_val = 360 + az_val
        
        if az_val > 360:
            az_val = az_val - 360

        r.append(r_val)
        az.append(az_val)
        el.append(el_val)

    return r, az, el

def read_measurements_from_csv(file_path):
    measurements = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if exists
        for row in reader:
            mr = float(row[7])  # MR column
            ma = float(row[8])  # MA column
            me = float(row[9])  # ME column
            mt = float(row[10])  # MT column
            x, y, z = sph2cart(ma, me, mr)  # Convert spherical to Cartesian coordinates
            measurements.append((x, y, z, mt))
    return measurements

def group_measurements_into_tracks(measurements):
    tracks = []
    used_indices = set()
    for i, (x_base, y_base, z_base, mt_base) in enumerate(measurements):
        if i in used_indices:
            continue
        track = [(x_base, y_base, z_base, mt_base)]
        used_indices.add(i)
        for j, (x, y, z, mt) in enumerate(measurements):
            if j in used_indices:
                continue
            if abs(mt - mt_base) < 0.050:
                track.append((x, y, z, mt))
                used_indices.add(j)
        tracks.append(track)
    return tracks

def is_valid_hypothesis(hypothesis):
    non_zero_hypothesis = [val for _, val in hypothesis if val != -1]
    return len(non_zero_hypothesis) == len(set(non_zero_hypothesis)) and len(non_zero_hypothesis) > 0

state_dim = 3
chi2_threshold = chi2.ppf(0.95, df=state_dim)

def mahalanobis_distance(x, y, cov_inv):
    delta = np.array(y)[:3] - np.array(x)[:3]
    return np.sqrt(np.dot(np.dot(delta, cov_inv), delta))


def perform_clustering_hypothesis_association(tracks, reports, cov_inv):
    clusters = []
    for report in reports:
        distances = [np.linalg.norm(np.array(track)[:3] - np.array(report)[:3]) for track in tracks]
        min_distance_idx = np.argmin(distances)
        if distances[min_distance_idx] < chi2_threshold:
            clusters.append([min_distance_idx])

    hypotheses = []
    for cluster in clusters:
        num_tracks = len(cluster)
        base = len(reports) + 1
        for count in range(base ** num_tracks):
            hypothesis = []
            for track_idx in cluster:
                report_idx = (count // (base ** track_idx)) % base
                hypothesis.append((track_idx, report_idx - 1))
            if is_valid_hypothesis(hypothesis):
                hypotheses.append(hypothesis)

    probabilities = calculate_probabilities(hypotheses, tracks, reports, cov_inv)

    max_associations, max_probs = find_max_associations(hypotheses, probabilities, reports)

    return max_associations, max_probs

def calculate_probabilities(hypotheses, tracks, reports, cov_inv):
    probabilities = []
    for hypothesis in hypotheses:
        prob = 1.0
        for track_idx, report_idx in hypothesis:
            if report_idx != -1:
                distance = mahalanobis_distance(tracks[track_idx], reports[report_idx], cov_inv)
                prob *= np.exp(-0.5 * distance ** 2)
        probabilities.append(prob)
    probabilities = np.array(probabilities)
    probabilities /= probabilities.sum()
    return probabilities

def find_max_associations(hypotheses, probabilities, reports):
    max_associations = [-1] * len(reports)
    max_probs = [0.0] * len(reports)
    for hypothesis, prob in zip(hypotheses, probabilities):
        for track_idx, report_idx in hypothesis:
            if report_idx != -1 and prob > max_probs[report_idx]:
                max_probs[report_idx] = prob
                max_associations[report_idx] = track_idx
    return max_associations, max_probs

def main():
    kalman_filter = CVFilter()
    csv_file_path = 'ttk_84.csv'
    measurements = read_measurements_from_csv(csv_file_path)

    csv_file_predicted = "ttk_84.csv"
    df_predicted = pd.read_csv(csv_file_predicted)
    filtered_values_csv = df_predicted[['F_TIM', 'F_X', 'F_Y', 'F_Z']].values

    A = cart2sph2(filtered_values_csv[:, 1], filtered_values_csv[:, 2], filtered_values_csv[:, 3])

    number = 1000
    result = np.divide(A[0], number)
    tracks = group_measurements_into_tracks(measurements)
    cov_inv = np.linalg.inv(np.eye(state_dim))

    for group_idx, group in enumerate(tracks):
        time_list = []
        r_list = []
        az_list = []
        el_list = []
        
        if len(group) == 0:
            continue

        print(f"Processing group {group_idx + 1}")

        initial_state = group[0]
        kalman_filter.initialize_filter_state(initial_state[0], initial_state[1], initial_state[2], 0, 0, 0, initial_state[3])

        for i, measurement in enumerate(group[1:]):
            current_time = measurement[3]
            kalman_filter.predict_step(current_time)
            kalman_filter.initialize_measurement_for_filtering(measurement[0], measurement[1], measurement[2], current_time)
            kalman_filter.update_step()

        associations, probabilities = perform_clustering_hypothesis_association(group, measurements, cov_inv)
        print(f"Associations: {associations}")
        print(f"Probabilities: {probabilities}")

        # Plotting
        plt.figure(figsize=(12, 6))
        plt.subplot(facecolor="white")
        plt.scatter(time_list, r_list, label='filtered range (code)', color='green', marker='*')
        plt.scatter(filtered_values_csv[:, 0], result, label='filtered range (track id 31)', color='red', marker='*')
        plt.xlabel('Time', color='black')
        plt.ylabel('Range (r)', color='black')
        plt.title('Range vs. Time', color='black')
        plt.grid(color='gray', linestyle='--')
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.subplot(facecolor="white")
        plt.scatter(time_list, az_list, label='filtered azimuth (code)', color='green', marker='*')
        plt.scatter(filtered_values_csv[:, 0], A[1], label='filtered azimuth (track id 31)', color='red', marker='*')
        plt.xlabel('Time', color='black')
        plt.ylabel('Azimuth (az)', color='black')
        plt.title('Azimuth vs. Time', color='black')
        plt.grid(color='gray', linestyle='--')
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.subplot(facecolor="white")
        plt.scatter(time_list, el_list, label='filtered elevation (code)', color='green', marker='*')
        plt.scatter(filtered_values_csv[:, 0], A[2], label='filtered elevation (track id 31)', color='red', marker='*')
        plt.xlabel('Time', color='black')
        plt.ylabel('Elevation (el)', color='black')
        plt.title('Elevation vs. Time', color='black')
        plt.grid(color='gray', linestyle='--')
        plt.legend()
        plt.tight_layout()
        plt.show()

    # plt.plot(result)
    # plt.xlabel('Index')
    # plt.ylabel('Value')
    # plt.title('Filtered Values')
    # plt.show()

if __name__ == '__main__':
    main()
