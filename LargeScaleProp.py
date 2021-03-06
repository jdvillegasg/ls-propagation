# Version: 1.0
# Author: JDVG

import numpy as np
import scipy.io as sio
from pathlib import Path
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.linear_model import LinearRegression
import pyproj as proj
from scipy.spatial.distance import cdist
from scipy import stats, interpolate
from sklearn.cluster import  KMeans
from scipy import optimize
import re

class LargeScaleProp:
    routes_dict = {"MKT": {"StBarbe": [9], "ParkingLot": [1, 2, 3, 4], "MarieCurie": [5, 6, 7, 8]},
                   "MXW": {"StBarbe": [1, 2], "ParkingLot": [9, 10], "MarieCurie": [6, 7, 8]}}

    kml_files_dict = {"StBarbe": [1, 2, 3, 4], "MarieCurie": [5, 6, 7, 8], "ParkingLot": [9, 10, 11, 12]}

    #rx = {"MXW": {"lat": 50.668627, "lon": 4.623663}, "MKT": {"lat": 50.669280, "lon": 4.620146}}
    #rx = {'MXW': {'x': 441.04262448035297, 'y': 150.15144391544163}, 'MKT': {'x': 192.17165641408064, 'y': 221.93809901550412}}
    rx = {'MXW': {'x': 309.7497587, 'y': 150.1859}, 'MKT': {'x': 60.878699, 'y': 221.972495}}

    firf = 30.2
    fc = 3.8e9
    vel_txc = 1.5

    def __init__(self, base_addr, BS, route_name, lmd=None):
        self.base_addr = base_addr
        self.BS = BS
        self.route_name = route_name
        self.lmd = lmd

        # Load the building coordinates
        matfile = sio.loadmat(r"C:\Users\julia\OneDrive - UCL\Pièces jointes\MATLAB\building_lines_all_considered.mat")
        tmp = matfile['buildings']

        erase_rows = np.r_[np.arange(37-1, 58), np.arange(59-1, 184), np.arange(185-1, 214), np.arange(333-1, 340)]
        tmp = np.delete(tmp, erase_rows, 0)

        st_pt = self.convert_DDDMS_to_planar(4326, 31370, tmp[:, 0], tmp[:, 2])
        fn_pt = self.convert_DDDMS_to_planar(4326, 31370, tmp[:, 1], tmp[:, 3])

        #rx_pt = self.convert_DDDMS_to_planar(4326, 31370, self.rx[BS]["lon"], self.rx[BS]["lat"])
        #print(rx_pt)

        # The minimum y value is given by the Parking Lot route. From executing the following snippet of code
        # we end up with the number in the offset dictionary.
        #
        # miny = []
        # for i in np.arange(start=0, stop=10):
        #     file_path = r"C:\Users\julia\Documents\TXCOORD\Mercator\t000000" + str(i) + ".gps"
        #     tx, info = obj_handle.get_coordinates_from_file(file_path)
        #     miny.append(tx['y'].min())

        self.offset_scenario = {}
        self.offset_scenario['x'] = min([st_pt['x'].min(), fn_pt['x'].min()])
        self.offset_scenario['y'] = min([st_pt['y'].min(), fn_pt['y'].min(), 150771.95911315177])

        #rx_pt['x'] = rx_pt['x'] - self.offset_scenario['x']
        #rx_pt['y'] = rx_pt['y'] - self.offset_scenario['y']
        #print(rx_pt)

        st_pt['x'] = st_pt['x'] - self.offset_scenario['x']
        fn_pt['x'] = fn_pt['x'] - self.offset_scenario['x']
        st_pt['y'] = st_pt['y'] - self.offset_scenario['y']
        fn_pt['y'] = fn_pt['y'] - self.offset_scenario['y']

        # Dictionary with starting and ending points of each line of each building in the scenario
        self.buildings = np.vstack((st_pt['x'], fn_pt['x'], st_pt['y'], fn_pt['y'])).T

    def get_all_rx_power_this_route(self, BS, route_name, dB=None):
        "Return a dictionary with all the RX Power from BS and route_name"
        route_number = self.routes_dict[BS][route_name]

        bs_rx_pow = {}

        for file in os.listdir(self.base_addr):
            if BS in file:
                usefull_filename = file.split('.')
                usefull_filename = usefull_filename[0]

                # Get route number, from the fact it is an integer
                tmp = [int(s) for s in usefull_filename.split('_') if s.isdigit()]
                route_number_this_file = tmp[0]

                if route_number_this_file in route_number:
                    mat_data = sio.loadmat(Path(self.base_addr).joinpath(file))
                    mat_data = mat_data["rx_power_" + BS.lower()]
                    mat_data = mat_data.flatten()

                    if dB is not None:
                        bs_rx_pow[route_name + str(route_number_this_file)] = 10*np.log10(mat_data.flatten())
                    else:
                        bs_rx_pow[route_name + str(route_number_this_file)] = mat_data.flatten()

        return bs_rx_pow

    def get_avg_all_rx_power_this_route(self, **kwargs):
        """Compute the ensemble average of all the rx power vectors available for a given route.
        First, reshaping of the different rx power vectors must be done.
        A simple approach is followed here:
         1. Set the size of all rx power vectors to the size of the minimum size vector.
         2. Drop samples each floor ( Ni/(Ni - Nmin) ).
         3. If dropped enough samples way before the end of a vector, stop droping samples.

         Usage:
         With dictionary
         rxp_mkt_mcurie = obj_handle.get_all_rx_power_this_route('MKT', 'MarieCurie')
         rxp = obj_handle.get_avg_all_rx_power_this_route(dict=rxp_mkt_mcurie)

         Without dictionary
         rxp = obj_handle.get_avg_all_rx_power_this_route(BS='MKT', routename='MarieCurie')
         """

        rxp_cutted = {}
        if 'dict' in kwargs:
            # User can provide the dictionary with the measurements, using the input keyword 'dict'
            rx_pows = kwargs['dict']
        else:
            rx_pows = self.get_all_rx_power_this_route(kwargs['BS'], kwargs['routename'])

        if len(rx_pows) > 1:
            min_sz_meas = min([it.shape[0] for a, it in rx_pows.items()])

            for route_name, rxp in rx_pows.items():
                rxp_sz = rxp.shape[0]

                if rxp.shape[0] != min_sz_meas:
                    discard_each_this = int(np.floor(rxp_sz / (rxp_sz - min_sz_meas)))
                    idx_del = np.arange(start=discard_each_this, step=discard_each_this,
                                        stop=discard_each_this * (rxp_sz - min_sz_meas) + 1)
                    rxp_cutted[route_name] = np.delete(rxp, idx_del)
        else:
            rxp_cutted = rx_pows

        df = pd.DataFrame(rxp_cutted)
        df_avg = df.mean(axis=1)

        return df_avg

    def empirical_distance_path_loss(self, rxp_dBm, Pt_dBm, tx, rx, FI):
        """Pr[W]/Pt[W] = 1/L ---->  Pr[W] = Pt[W]/L \
        1/L can be thought as an attenuator factor. Distance-based model assumes 1/L,
        the attenuating factor, is proportional to the inverse of the distance to some
        power.
        That is 1/L = k/r^n, where k is a constant, r is the distance [m] and n is the
        path loss exponent [dimensionless].
        This gives L[dB] = 10n log(r) - 10 log(k) ---> L[dB] = 10n log(r[m]) + K, where K
        is a constant.

        10 log(Pt[W]/Pr[W]) = L[dB] ---> L[dB] = 10 log(Pt[W]) - 10 log(Pr[W])

        L[dB] = Pt[dBW] - Pr[dBW] = (30 + Pt[dBm]) - (30 + Pr[dBm]) = Pt[dBm] - Pr[dBm]
        """

        # Friis equation at 1 m
        try:
            L_1m = 10*np.log10( np.square(4 * np.pi / self.lmd) )
        except NameError:
            print('ERROR: provide wavelength when creating an instance of class LargeScaleProp')

        L = Pt_dBm - rxp_dBm
        #L = L - L_1m

        # Smooth-out the small-scale fading
        dist_wavs = 10
        samples_to_avg = int(np.floor(((dist_wavs*3e8)/(self.vel_txc*self.fc)) * self.firf))

        df_L = pd.Series(L)
        L_smoothed = df_L.rolling(samples_to_avg).mean().to_numpy()
        L_smoothed[:samples_to_avg-1] = L[:samples_to_avg-1]

        r = cdist(tx, rx, metric='euclidean')
        idx_r_sorted = np.argsort(r, axis=0).flatten()

        reg_indp_var = 10 * np.log10(r[idx_r_sorted])
        reg_dep_var = L_smoothed[idx_r_sorted]

        if FI:
            fit_reg = LinearRegression()
            fit_reg.fit(reg_indp_var, reg_dep_var)
        else:
            fit_reg = LinearRegression(fit_intercept=False)
            fit_reg.fit(reg_indp_var, reg_dep_var - L_1m)

        sh = reg_dep_var - fit_reg.predict(reg_indp_var)
        sh = sh[np.argsort(idx_r_sorted)]

        return fit_reg, reg_indp_var, reg_dep_var, sh

    def prune_trajectories(self, tx, BS, gps_file_num):
        """Prunes the trajectories followed by the TX. Under the assumption that some of the GPS data
        behaves randomly at the beginning of the trajectory, we drop some of the coordinates samples.
        The samples to be dropped had been found manually (visually) for each .kml file."""

        switcher = {'MKT': {0: [],
                            1: np.concatenate((np.arange(start=4, stop=7), np.arange(start=167, stop=215))),
                            2: np.concatenate((np.arange(start=0, stop=23), np.arange(start=190, stop=213))),
                            3: np.concatenate((np.arange(start=0, stop=47), np.arange(start=181, stop=225))),
                            4: np.concatenate((np.arange(start=0, stop=14), np.arange(start=161, stop=210))),
                            5: np.concatenate((np.arange(start=0, stop=14), np.arange(start=192, stop=195))),
                            6: [],
                            7: [],
                            8: np.concatenate((np.arange(start=0, stop=14), np.arange(start=163, stop=182))),
                            9: [],
                            10:[]},
                    'MXW': {0: [],
                            1: np.concatenate((np.arange(start=0, stop=58), np.arange(start=238, stop=283))),
                            2: np.concatenate((np.arange(start=0, stop=12), np.arange(start=204, stop=227))),
                            3: np.concatenate((np.arange(start=0, stop=13), np.arange(start=201, stop=242))),
                            4: np.concatenate((np.arange(start=0, stop=20), np.arange(start=212, stop=246))),
                            5: np.concatenate((np.arange(start=0, stop=27), np.arange(start=170, stop=196))),
                            6: np.concatenate((np.arange(start=0, stop=19), np.arange(start=150, stop=176))),
                            7: np.concatenate((np.arange(start=0, stop=14), np.arange(start=147, stop=173))),
                            8: np.concatenate((np.arange(start=0, stop=19), np.arange(start=142, stop=203))),
                            9: np.concatenate((np.arange(start=0, stop=39), np.arange(start=151, stop=181))),
                            10: np.concatenate((np.arange(start=0, stop=17), np.arange(start=132, stop=163))),
                            11: np.concatenate((np.arange(start=0, stop=18), np.arange(start=66, stop=71), np.arange(start=138, stop=171))),
                            12: np.concatenate((np.arange(start=0, stop=14), np.arange(start=132, stop=164)))}}

        tx_pruned = {}

        tx_pruned['x'] = np.delete(tx['x'], switcher[BS][gps_file_num])
        tx_pruned['y'] = np.delete(tx['y'], switcher[BS][gps_file_num])

        return tx_pruned

    def get_coordinates_from_file(self, file_path, offset=None):
        """Get TX coordinates from the .gps files. Treat the files as .csv files with certain structure.
        Since Measurement campaign was carried out in Belgium we know, the 3-digits longitude field in the field
        is used as follows: first digit is the degree, and second-to-third digits are the minutes (in DMS).
        Since Belgium is close to Greenwich it can't be the case that the first-to-second digits refer to the
        degrees in the DMS system.
        """
        tx_info = {}
        tx = {}

        df = pd.read_csv(file_path, sep=' :', names=['GPS', 'TAG', 'QUALFIX', 'SAT', 'UTC', 'LAT', 'LON', 'ALT'])
        df.drop(columns=['GPS', 'QUALFIX', 'SAT'])

        # Process ALT column
        try:
            tx_info['ALT'] = np.asarray([float(val[1][1:-1]) for key, val in df.ALT.str.split(':').to_dict().items()])
        except:
            print('WARN: ALT is NONE')

        # Process LON column
        tmp = [val[1][1:-3] for key, val in df.LON.str.split(':').to_dict().items()]
        tx_info['LON'] = np.asarray([float(i[2]) + float(i[3:]) / 60 for i in tmp])

        # Process LAT column
        tmp = [val[1][1:-3] for key, val in df.LAT.str.split(':').to_dict().items()]
        tx_info['LAT'] = np.asarray([float(i[0:2]) + float(i[2:]) / 60 for i in tmp])

        # Process UTC column
        try:
            tx_info['UTC'] = np.asarray([float(val[1][1:-1]) for key, val in df.UTC.str.split(':').to_dict().items()])
        except:
            print('WARN: UTC is NONE')

        # Process TAG column: save it just in case --->> not known its functionality
        try:
            tx_info['TAG'] = np.asarray([int(val[1]) for key, val in df.TAG.str.split(':').to_dict().items()])
        except:
            print('WARN: TAG is NONE')

        tx = self.convert_DDDMS_to_planar(4326, 31370, tx_info['LON'], tx_info['LAT'], offset)

        return tx, tx_info

    def interpolate_coordinates(self, txc, nCycles, cycleRate):
        """Interpolate transceiver coordinates based on the distance between available
        transceiver coordinates."""

        txc = np.array([txc['x'].tolist(), txc['y'].tolist()]).T
        txc_rotated = np.concatenate((txc[1:, :], txc[0, :][np.newaxis]))

        adjacent_distances = np.sqrt(np.sum(np.square(txc - txc_rotated), axis=1))

        vel_txc = (cycleRate/nCycles)*np.sum(adjacent_distances)
        self.vel_txc = vel_txc

        n_irfs_between_coords = np.floor((cycleRate / vel_txc) * adjacent_distances)
        samples_to_add = int(nCycles - np.sum(n_irfs_between_coords))

        # Sort distances in decreasing order
        idx_distances = np.fliplr(np.argsort(adjacent_distances)[np.newaxis])[0]

        n_irfs_between_coords[idx_distances[0:samples_to_add]] = n_irfs_between_coords[idx_distances[0:samples_to_add]] + 1
        n_irfs_between_coords = n_irfs_between_coords - 1

        coordinates_interpolated = []

        # Possibly sub-optimal python coding (translated from MATLAB)
        for idx, tx_i in enumerate(txc):
            coordinates_interpolated.append(tx_i.tolist())

            route_pointing_vector = txc_rotated[idx, :] - tx_i
            unitary_route_pointing_vector = route_pointing_vector/np.linalg.norm(route_pointing_vector)

            step_intermediate_points = adjacent_distances[idx] / n_irfs_between_coords[idx]

            for k in np.arange(start=1, stop=n_irfs_between_coords[idx] + 1, step=1):
                coordinates_interpolated.append((tx_i + k*step_intermediate_points*unitary_route_pointing_vector).tolist())

        coordinates_interpolated = np.asarray(coordinates_interpolated)

        nCycles_Comp = coordinates_interpolated.shape[0]

        # Just  drop first coordinates if bigger size
        if nCycles_Comp > nCycles:
            coordinates_interpolated = np.delete(coordinates_interpolated, np.arange(start=0, stop=nCycles_Comp - nCycles, step=1), axis=0)

        return coordinates_interpolated

    def convert_DDDMS_to_planar(self, epsg_in, epsg_out, input_lon, input_lat, offset=None):
        """A wrapper to pyproj methods. DD stands for Decimal degrees, while DMS stands for
         Degrees, Minutes, Seconds format"""

        txc = {}

        # setup your projections, assuming you're using WGS84 geographic
        crs_wgs = proj.Proj(init='epsg:' + str(epsg_in))
        crs_bng = proj.Proj(init='epsg:' + str(epsg_out))  # use the Belgium epsg code

        # then cast your geographic coordinate pair to the projected system
        txc['x'], txc['y'] = proj.transform(crs_wgs, crs_bng, input_lon, input_lat)

        # Remove offset
        if offset is not None:
            txc['x'] = txc['x'] - offset['x']
            txc['y'] = txc['y'] - offset['y']

        return txc

    def get_snapshot_partition_LoS_NLoS_from_delay(self, ISIS_ADDR, maxNumWaves, tx, rx):
        """Partition the snapshots based on the MPC delays. Only available for measurements
        for which the delays as computed from SAGE (or related) are available."""

        mpc_taus = pd.read_csv(ISIS_ADDR, delimiter=',', header=None, usecols=np.arange(0, maxNumWaves).tolist())
        mpc_taus = mpc_taus * 5e-9

        max_mpc_taus = mpc_taus.max(axis=1).to_numpy()
        min_mpc_taus = mpc_taus.min(axis=1).to_numpy()

        LoS_tau = cdist(tx, rx, metric='euclidean')/3e8

        if LoS_tau.shape[0] != max_mpc_taus.shape[0]:
            print("ERROR: The extracted delays don't correspond to the route: mismatch in dimensions")
            return

        yerrs = [min_mpc_taus, max_mpc_taus]

        plt.errorbar(np.arange(1, tx.shape[0]+1), LoS_tau, yerr=yerrs, fmt='o')
        plt.show()

    def get_snapshot_partition_LoS_NLoS_from_rxp(self, rxp_dBm, dB=None):
        """Partition the snapshots based on received power. """

        if dB is not None:
            rxp = pow(10, rxp_dBm/10)
        else:
            rxp = rxp_dBm

        k_factor = rxp/np.min(rxp)

        v_quantiles = [0.25, 0.5, 0.75, 0.9]
        quantiles_k_factor = np.quantile(k_factor, v_quantiles)

        quantiles_k_factor = np.insert(quantiles_k_factor, 0, 0)

        rxp_grouped_by_quantile = []
        idx_group_members = []
        for i in np.arange(0, quantiles_k_factor.shape[0]-1):
            rxp_grouped_by_quantile.append(
                [j for j in k_factor if (j >= quantiles_k_factor[i]) & (j < quantiles_k_factor[i+1])])
            idx_group_members.append(
                [k_factor.tolist().index(j) for j in k_factor if (j >= quantiles_k_factor[i]) & (j < quantiles_k_factor[i + 1])])

        rxp_grouped_by_quantile.append(
            [j for j in k_factor if j >= quantiles_k_factor[i+1]])
        idx_group_members.append(
            [k_factor.tolist().index(j) for j in k_factor if
             j >= quantiles_k_factor[i+1]])

        return rxp_grouped_by_quantile, idx_group_members, quantiles_k_factor

    def best_gof_distr(self, signal, list_of_dists):
        results = []
        for i in list_of_dists:
            dist = getattr(stats, i)
            param = dist.fit(signal)
            a = stats.kstest(signal, i, args=param)
            results.append([i, a[0], a[1], param])

        results.sort(key=lambda x: x[2], reverse=True)

        return results

    def get_snaps_same_space_diff_meas(self, txs, k):
        """For a given route there were taken different number of measurements. Each measurement
        might have different number of samples. However, the locations of the mobile terminal might be similar,
        since all measurements refered to the same route.
        This function identifies the snapshots in each measurement, that belong to the same physical space."""

        kmeans = KMeans(n_clusters=k)
        kmeans.fit(txs)
        km_pred = kmeans.predict(txs)

        plt.scatter(txs[:, 0], txs[:, 1], c=km_pred)
        plt.show()

        print(km_pred)

    def utility_get_gps_file_path(self, bs, file_number, cnt):
        if bs == 'MKT':
            if file_number[cnt] > 9:
                file_path = r"C:\Users\julia\Documents\TXCOORD\Mercator\t00000" + str(file_number[cnt]) + ".gps"
            else:
                file_path = r"C:\Users\julia\Documents\TXCOORD\Mercator\t000000" + str(file_number[cnt]) + ".gps"
        else:
            if file_number[cnt] > 9:
                file_path = r"C:\Users\julia\Documents\TXCOORD\Maxwell\t00000" + str(file_number[cnt]) + ".gps"
            else:
                file_path = r"C:\Users\julia\Documents\TXCOORD\Maxwell\t000000" + str(file_number[cnt]) + ".gps"

        return file_path

    def segments_fit(self, X, Y, count):
        """
        Approximate a curve by a piece-wise line, with a configurable number of segments
        :param X:
        :param Y:
        :param count: number of segments
        :return:
        """
        xmin = X.min()
        xmax = X.max()

        seg = np.full(count - 1, (xmax - xmin) / count)

        px_init = np.r_[np.r_[xmin, seg].cumsum(), xmax]
        py_init = np.array([Y[np.abs(X - x) < (xmax - xmin) * 0.01].mean() for x in px_init])

        def func(p):
            seg = p[:count - 1]
            py = p[count - 1:]
            px = np.r_[np.r_[xmin, seg].cumsum(), xmax]
            return px, py

        def err(p):
            px, py = func(p)
            Y2 = np.interp(X, px, py)
            return np.mean((Y - Y2) ** 2)

        r = optimize.minimize(err, x0=np.r_[seg, py_init], method='Nelder-Mead')
        return func(r.x)

class LsPropPlots(LargeScaleProp):
    def __init__(self, dict_what_plt, init_col=3, init_rows=3, bs_plt='MXW'):
        """

        :param dict_what_plt: dictionary containing the fields to plot:
                              -RXP
                              -PL
                              -Sh-Fit
                              -Sh-QQ
                              -KFactor-Route
                              -Stat-Regs
        :param init_col:
        :param init_rows:
        :param bs_plt:
        """

        #BS
        self.bs_plt = bs_plt

        # Says what to plot
        self.dict_what_plt = dict_what_plt

        # Define number of subplots in a figure
        self.subplts_matrix = {'MKT': {'rows': 3, 'cols': 3}, 'MXW': {'rows': 3, 'cols': 3}}
        self.idx_subplt_leftover = {'MKT': [], 'MXW': {'rows': [2], 'cols': [1, 2]}}

        # Assign to each route a place in the figure plot
        self.subplots_mapp = {'MXW': {'ParkingLot10': (0, 1), 'ParkingLot9': (0, 0),
                                      'MarieCurie6': (0, 2), 'MarieCurie7': (1, 0), 'MarieCurie8': (1, 1),
                                      'StBarbe1': (1, 2), 'StBarbe2': (2, 0)},
                              'MKT': {'ParkingLot1': (0, 0), 'ParkingLot2': (0, 1), 'ParkingLot3': (0, 2),
                                      'ParkingLot4': (1, 0), 'MarieCurie5': (1, 1), 'MarieCurie6': (1, 2),
                                      'MarieCurie7': (2, 0), 'MarieCurie8': (2, 1), 'StBarbe9': (2, 2)}}

        # Create the figures and subplot axes for the plots we want to do
        self.list_figs = {}
        for name_sbplt, en_subplt in self.dict_what_plt.items():
            if en_subplt:
                self.list_figs[name_sbplt] = plt.subplots(self.subplts_matrix[bs_plt]['rows'], self.subplts_matrix[bs_plt]['cols'])
                if self.idx_subplt_leftover[bs_plt]:
                    for i in self.idx_subplt_leftover[bs_plt]['rows']:
                        for j in self.idx_subplt_leftover[bs_plt]['cols']:
                            self.list_figs[name_sbplt][1][i, j].set_visible(False) # Choosing the last figure axes subplot i,j

    def right_idx_format(self, i1, i2):
        """
        Indexes for subplots based based on i1, i2. Both i1, i2 can be empty
        :param i1:
        :param i2:
        :return:
        """
        # Not allowed this option
        # if (i1 =='') & (i2 ==''):
        #    ri =

        if (i1 == '') & (i2 != ''):
            ri = i2
        if (i1 != '') & (i2 == ''):
            ri = i1
        if (i1 != '') & (i2 != ''):
            ri = (i1, i2)

        return ri

    def plot_fig_rxpowers(self, plt_idx_row, plt_idx_col, sample_number, rx_p):
        """
        Plot the received power
        :param plt_idx_row:
        :param plt_idx_col:
        :param sample_number:
        :param rx_p: dictionary
        :return:
        """
        ri = self.right_idx_format(plt_idx_row, plt_idx_col)

        n_x_ticks = 3
        dyn_range_x = sample_number.shape[0]

        n_y_ticks = 3
        dyn_range_y = np.max(rx_p) - np.min(rx_p)

        N = 1000
        cumsum = np.cumsum(np.insert(rx_p, 0, 0))
        rxp_smooth = (cumsum[N:] - cumsum[:-N]) / float(N)
        rxp_smooth2 = np.r_[rxp_smooth, rx_p[-(N - 1):]]

        self.list_figs['RXP'][1][ri].xaxis.set_major_locator(mpl.ticker.MultipleLocator(np.floor(dyn_range_x / n_x_ticks)))

        self.list_figs['RXP'][1][ri].set_xlim(1, rx_p.shape[0])
        self.list_figs['RXP'][1][ri].set_ylim(np.min(rx_p), np.max(rx_p))

        self.list_figs['RXP'][1][ri].yaxis.set_major_locator(mpl.ticker.MultipleLocator(np.floor(dyn_range_y / n_y_ticks)))
        self.list_figs['RXP'][1][ri].set_xlabel('Snapshot number', fontsize=30)
        self.list_figs['RXP'][1][ri].set_ylabel('$P_{RX}$ [dBm]', fontsize=30)

        self.list_figs['RXP'][1][ri].xaxis.set_tick_params(which='major', size=10, width=1.5, direction='in', top='on')
        self.list_figs['RXP'][1][ri].xaxis.set_tick_params(which='minor', size=10, width=1.5, direction='in', top='on')
        self.list_figs['RXP'][1][ri].yaxis.set_tick_params(which='major', size=10, width=1.5, direction='in', right='on')
        self.list_figs['RXP'][1][ri].yaxis.set_tick_params(which='minor', size=10, width=1.5, direction='in', right='on')

        self.list_figs['RXP'][1][ri].plot(sample_number, rx_p, ls='--', linewidth=1.2, alpha=0.7, color='b')
        self.list_figs['RXP'][1][ri].plot(np.arange(np.floor(N / 2), np.floor(N / 2) + rxp_smooth.shape[0]), rxp_smooth, ls='-.',
                           linewidth=3, color='r')

    def plot_fig_pathloss(self, plt_idx_row, plt_idx_col, regressor, predictor, reg, FI):
        """
        Plot the path loss
        :param plt_idx_row:
        :param plt_idx_col:
        :param regressor:
        :param predictor:
        :param reg:
        :param FI: floating intercept (True) or closed-in intercept (False)
        :return:
        """
        ri = self.right_idx_format(plt_idx_row, plt_idx_col)

        d_tx_rx = pow(10, regressor / 10)

        # aux = np.arange(start=1, stop=np.max(d_tx_rx)+1, step=1)
        aux = np.arange(start=np.min(d_tx_rx), stop=np.max(d_tx_rx) + 1, step=1)
        reg_indp_var_interp = aux.reshape(aux.shape[0], 1)

        if FI:
            pred_L = reg.predict(10 * np.log10(reg_indp_var_interp))
            str_FI = 'FI'
        else:
            pred_L = reg.predict(10 * np.log10(reg_indp_var_interp)) + 10 * np.log10(
                np.square(4 * np.pi / super().lmd))
            str_FI = 'CI'

        n_x_ticks = 3
        dyn_range_x = np.max(d_tx_rx) - 1
        # ax_pklot2[ri].xaxis.set_major_locator(mpl.ticker.MultipleLocator(np.floor(dyn_range_x / n_x_ticks)))

        n_y_ticks = 3
        dyn_range_y = np.abs(np.max(pred_L) - np.min(pred_L))
        # ax_pklot2[ri].yaxis.set_major_locator(mpl.ticker.MultipleLocator(np.floor(dyn_range_y/n_y_ticks)))

        self.list_figs['PL'][1][ri].set_xlabel('Transceiver distance [m]', fontsize=30)
        self.list_figs['PL'][1][ri].set_ylabel('L [dB]', fontsize=30)

        self.list_figs['PL'][1][ri].xaxis.set_tick_params(which='major', size=10, width=1.5, direction='in', top='on')
        self.list_figs['PL'][1][ri].xaxis.set_tick_params(which='minor', size=10, width=1.5, direction='in', top='on')
        self.list_figs['PL'][1][ri].yaxis.set_tick_params(which='major', size=10, width=1.5, direction='in', right='on')
        self.list_figs['PL'][1][ri].yaxis.set_tick_params(which='minor', size=10, width=1.5, direction='in', right='on')

        self.list_figs['PL'][1][ri].plot(d_tx_rx, predictor, color='b', ls='', marker='x', markersize=5)
        self.list_figs['PL'][1][ri].plot(reg_indp_var_interp, pred_L, color='k', linewidth=4)
        # ax_pklot2[ri].semilogx(d_tx_rx, predictor, color='b', ls='', marker='x', markersize=3)
        # ax_pklot2[ri].semilogx(reg_indp_var_interp, pred_L, color='k', linewidth=5)

        # ax_pklot2[ri].legend(labels=['Path Loss', 'n=' + str(np.around(reg.coef_[0], 2))],
        # bbox_to_anchor=(1, 0), loc=4, frameon=False, fontsize=16)

        ploss_exp = np.around(reg.coef_[0], 2)
        print('Path Loss Exponent' + str_FI + ': ', ploss_exp)

    def plot_fig_distribution_fits(self, plt_idx_row, plt_idx_col, signal, alpha, title, list_of_dists_reduced):
        """Fit a gaussian distribution and BGoF to the shadowing signal and plot it"""
        ri = self.right_idx_format(plt_idx_row, plt_idx_col)

        # Set up axes for plotting
        self.list_figs['Sh-Fit'][1][ri].set_xlabel(title, fontsize=30)
        self.list_figs['Sh-Fit'][1][ri].set_ylabel('Density', fontsize=30)

        self.list_figs['Sh-Fit'][1][ri].xaxis.set_tick_params(which='major', size=10, width=1.5, direction='in', top='on')
        self.list_figs['Sh-Fit'][1][ri].xaxis.set_tick_params(which='minor', size=10, width=1.5, direction='in', top='on')
        self.list_figs['Sh-Fit'][1][ri].yaxis.set_tick_params(which='major', size=10, width=1.5, direction='in', right='on')
        self.list_figs['Sh-Fit'][1][ri].yaxis.set_tick_params(which='minor', size=10, width=1.5, direction='in', right='on')

        # Fit the shadowing
        norm = getattr(stats, 'norm')
        mean_g, std_g = norm.fit(signal)
        parameters = norm.fit(signal)
        x = np.linspace(np.min(signal), np.max(signal), 1000)
        pdf_gaussian = norm.pdf(x, mean_g, std_g)

        # Kolmogorov-Smirnov test
        ks_test = stats.kstest(signal, 'norm', args=parameters)
        critical_value = stats.ksone.ppf(1 - alpha / 2, signal.shape[0])

        # BGF
        fitted_distr = super().best_gof_distr(signal, list_of_dists_reduced)
        best_gof_dist = getattr(stats, fitted_distr[0][0])
        pdf_best_gof = best_gof_dist.pdf(x, *fitted_distr[0][3])

        self.list_figs['Sh-Fit'][1][ri].hist(signal, density=True)
        self.list_figs['Sh-Fit'][1][ri].plot(x, pdf_gaussian, color='k', linewidth=1.5)
        self.list_figs['Sh-Fit'][1][ri].plot(x, pdf_best_gof, color='r', linewidth=1.5)
        # ax_pklot[ri].legend(labels=['$\mathcal{N}$', 'BGF'], bbox_to_anchor=(0, 1), loc=2, frameon=False, fontsize=20)

        print('Sh. Gauss. KS fit statistic:', ks_test)
        print('Sh. Gauss. KS fit critical value:', critical_value)

        print('Sh. gauss mean: ', mean_g)
        print('Sh. gauss std: ', std_g)

        print(title + ' BGoF KS fit statistic:', fitted_distr[0][1])
        print(title + ' BGoF KS fit critical value:', critical_value)

        print(title + ' BGoF KS fit distr. name:', fitted_distr[0][0])
        print(title + ' BGoF KS fit distr. parameters:', fitted_distr[0][3])

        return mean_g, std_g, fitted_distr

    def plot_qq_fig(self, plt_idx_row, plt_idx_col, sh):
        ri = self.right_idx_format(plt_idx_row, plt_idx_col)

        self.list_figs['Sh-QQ'][1][ri].xaxis.set_tick_params(which='major', size=10, width=1.5, direction='in', top='on')
        self.list_figs['Sh-QQ'][1][ri].xaxis.set_tick_params(which='minor', size=10, width=1.5, direction='in', top='on')
        self.list_figs['Sh-QQ'][1][ri].yaxis.set_tick_params(which='major', size=10, width=1.5, direction='in', right='on')
        self.list_figs['Sh-QQ'][1][ri].yaxis.set_tick_params(which='minor', size=10, width=1.5, direction='in', right='on')

        stats.probplot(sh, dist='norm', plot=self.list_figs['Sh-Fit'][1][ri])

    def plot_k_factor_chosen_quantiles(self, plt_idx_row, plt_idx_col, tx, idxs, bs):
        ri = self.right_idx_format(plt_idx_row, plt_idx_col)

        self.list_figs['KFactor-Route'][1][ri].xaxis.set_major_locator(mpl.ticker.MultipleLocator(50))
        self.list_figs['KFactor-Route'][1][ri].yaxis.set_major_locator(mpl.ticker.MultipleLocator(50))

        self.list_figs['KFactor-Route'][1][ri].xaxis.set_tick_params(which='major', size=10, width=1.5, direction='in', top='on')
        self.list_figs['KFactor-Route'][1][ri].xaxis.set_tick_params(which='minor', size=10, width=1.5, direction='in', top='on')
        self.list_figs['KFactor-Route'][1][ri].yaxis.set_tick_params(which='minor', size=10, width=1.5, direction='in', right='on')
        self.list_figs['KFactor-Route'][1][ri].yaxis.set_tick_params(which='major', size=10, width=1.5, direction='in', right='on')

        super().plot_buildings_and_this_traj(tx, bs, ax=self.list_figs['KFactor-Route'][1][ri], idx=idxs)

    def plot_delay_spreads(self, plt_idx_row, plt_idx_col, alpha, cnt, bs, method, thr_level, route_name, list_of_dists_reduced):
        """Find out the BGoF distribution for the delay spreads"""
        ri = self.right_idx_format(plt_idx_row, plt_idx_col)

        self.list_figs['CDF-Del-Spreads'][1][ri].xaxis.set_tick_params(which='major', size=10, width=1.5, direction='in', top='on')
        self.list_figs['CDF-Del-Spreads'][1][ri].xaxis.set_tick_params(which='minor', size=10, width=1.5, direction='in', top='on')
        self.list_figs['CDF-Del-Spreads'][1][ri].yaxis.set_tick_params(which='minor', size=10, width=1.5, direction='in', right='on')
        self.list_figs['CDF-Del-Spreads'][1][ri].yaxis.set_tick_params(which='major', size=10, width=1.5, direction='in', right='on')

        dict_var_delay = {'MXW': {'StBarbe': [0, 1], 'MarieCurie': [2, 3, 4], 'ParkingLot': [5, 6]},
                          'MKT': {'ParkingLot': [0, 1, 2, 3], 'MarieCurie': [4, 5, 6, 7], 'StBarbe': [8]}}

        if bs == 'MKT':
            data = sio.loadmat(
                r'C:\Users\julia\OneDrive - UCL\Pièces jointes\MATLAB\var_delay_statistics_sta_regs_' + method + '_thr_' + str(
                    thr_level) + '_MKT.mat')
        elif bs == 'MXW':
            data = sio.loadmat(
                r'C:\Users\julia\OneDrive - UCL\Pièces jointes\MATLAB\var_delay_statistics_sta_regs_' + method + '_thr_' + str(
                    thr_level) + '_MXW.mat')

        ds = data['delay_spreads'][dict_var_delay[bs][route_name]]

        ds_avg = np.mean(ds[cnt][0], axis=1) * 1e9
        critical_value = stats.ksone.ppf(1 - alpha / 2, ds_avg.shape[0])
        fitted_distr_ds = super().best_gof_distr(ds_avg, list_of_dists_reduced)

        x_ds = np.linspace(np.min(ds_avg), np.max(ds_avg), 1000)
        best_gof_dist_ds = getattr(stats, fitted_distr_ds[0][0])
        cdf_best_gof_ds = best_gof_dist_ds.cdf(x_ds, *fitted_distr_ds[0][3])

        self.list_figs['CDF-Del-Spreads'][1][ri].hist(ds_avg, 50, density=True, cumulative=True, color='b', histtype='step')
        # self.list_figs['Sh-Fit'][1][ri].hist(ds_avg, 50, density=True, color='b')
        self.list_figs['CDF-Del-Spreads'][1][ri].plot(x_ds, cdf_best_gof_ds, color='b', linewidth=1.5, ls='--', label=str(cnt))
        self.list_figs['CDF-Del-Spreads'][1][ri].set_xlabel('Delay spreads [$\mu$s]')
        self.list_figs['CDF-Del-Spreads'][1][ri].set_ylabel('CDF')
        self.list_figs['CDF-Del-Spreads'][1][ri].legend(labels=['Fit: ' + fitted_distr_ds[0][0], 'Empirical'],
                            bbox_to_anchor=(0, 1), loc=2, frameon=False, fontsize=16)

        median_ds = np.quantile(ds_avg, 0.5)
        std_ds = np.std(ds_avg)

        print('Median Del. Spread Route ' + route_name + str(cnt) + ' : ' + str(median_ds))
        print('Std Dev Del. Spread Route ' + route_name + str(cnt) + ' : ' + str(std_ds))

        print('Del. Spreads KS fit statistic:', fitted_distr_ds[0][1])
        print('Del. Spreads KS fit critical value:', critical_value)

        print('Del. Spreads BGoF disitr. name: ', fitted_distr_ds[0][0])
        print('Del. Spreads KS fit distr. parameters:', fitted_distr_ds[0][3])

        return median_ds, std_ds, ds_avg

    def plot_delay_avgs(self, plt_idx_row, plt_idx_col, alpha, cnt, bs, method, thr_level, route_name, list_of_dists_reduced):
        """Find out the BGoF distribution for the delay avgs"""
        ri = self.right_idx_format(plt_idx_row, plt_idx_col)

        self.list_figs['CDF-Del-Avg'][1][ri].xaxis.set_tick_params(which='major', size=10, width=1.5, direction='in', top='on')
        self.list_figs['CDF-Del-Avg'][1][ri].xaxis.set_tick_params(which='minor', size=10, width=1.5, direction='in', top='on')
        self.list_figs['CDF-Del-Avg'][1][ri].yaxis.set_tick_params(which='minor', size=10, width=1.5, direction='in', right='on')
        self.list_figs['CDF-Del-Avg'][1][ri].yaxis.set_tick_params(which='major', size=10, width=1.5, direction='in', right='on')

        dict_var_delay = {'MXW': {'StBarbe': [0, 1], 'MarieCurie': [2, 3, 4], 'ParkingLot': [5, 6]},
                          'MKT': {'ParkingLot': [0, 1, 2, 3], 'MarieCurie': [4, 5, 6, 7], 'StBarbe': [8]}}

        if bs == 'MKT':
            data = sio.loadmat(
                r'C:\Users\julia\OneDrive - UCL\Pièces jointes\MATLAB\var_delay_statistics_sta_regs_' + method + '_thr_' + str(
                    thr_level) + '_MKT.mat')
        elif bs == 'MXW':
            data = sio.loadmat(
                r'C:\Users\julia\OneDrive - UCL\Pièces jointes\MATLAB\var_delay_statistics_sta_regs_' + method + '_thr_' + str(
                    thr_level) + '_MXW.mat')

        dm = data['delay_avgs'][dict_var_delay[bs][route_name]]

        dm_avg = np.mean(dm[cnt][0], axis=1) * 1e9
        critical_value = stats.ksone.ppf(1 - alpha / 2, dm_avg.shape[0])
        fitted_distr_dm = super().best_gof_distr(dm_avg, list_of_dists_reduced)

        x_dm = np.linspace(np.min(dm_avg), np.max(dm_avg), 1000)
        best_gof_dist_dm = getattr(stats, fitted_distr_dm[0][0])
        cdf_best_gof_dm = best_gof_dist_dm.cdf(x_dm, *fitted_distr_dm[0][3])

        self.list_figs['CDF-Del-Avg'][1][ri].hist(dm_avg, 50, density=True, cumulative=True, color='b', histtype='step')
        # self.list_figs['Sh-Fit'][1][ri].hist(dm_avg, 50, density=True, color='b')
        self.list_figs['CDF-Del-Avg'][1][ri].plot(x_dm, cdf_best_gof_dm, color='b', linewidth=1.5, ls='--', label=str(cnt))
        self.list_figs['CDF-Del-Avg'][1][ri].set_xlabel('Delay averages [$\mu$s]')
        self.list_figs['CDF-Del-Avg'][1][ri].set_ylabel('PDF')
        self.list_figs['CDF-Del-Avg'][1][ri].legend(labels=['Fit: ' + fitted_distr_dm[0][0], 'Empirical'],
                            bbox_to_anchor=(0, 1), loc=2, frameon=False, fontsize=16)

        median_dm = np.quantile(dm_avg, 0.5)
        std_dm = np.std(dm_avg)

        print('Median Del. Avgs Route ' + route_name + str(cnt) + ' : ' + str(median_dm))
        print('Std Dev Del. Avgs Route ' + route_name + str(cnt) + ' : ' + str(std_dm))

        print('Del. Avgs KS fit statistic:', fitted_distr_dm[0][1])
        print('Del. Avgs KS fit critical value:', critical_value)

        print('Del. Avgs BGoF disitr. name: ', fitted_distr_dm[0][0])
        print('Del. Avgs KS fit distr. parameters:', fitted_distr_dm[0][3])

    def plot_delay_spreads_route(self, ds):
        """
        FOR TESTING PURPOSES
        :param ds:
        :return:
        """
        ri = 1
        self.list_figs['Sh-Fit'][1][ri].xaxis.set_tick_params(which='major', size=10, width=1.5, direction='in', top='on')
        self.list_figs['Sh-Fit'][1][ri].xaxis.set_tick_params(which='minor', size=10, width=1.5, direction='in', top='on')
        self.list_figs['Sh-Fit'][1][ri].yaxis.set_tick_params(which='minor', size=10, width=1.5, direction='in', right='on')
        self.list_figs['Sh-Fit'][1][ri].yaxis.set_tick_params(which='major', size=10, width=1.5, direction='in', right='on')

        self.list_figs['Sh-Fit'][1][ri].plot(ds, marker='x', markersize=5)

    def plot_routes_and_buildings(self, plt_idx_row, plt_idx_col, tx, bs, buildings, rx):
        """Wrapper to plot_buildings_and_this_traj"""
        ri = self.right_idx_format(plt_idx_row, plt_idx_col)
        self.plot_buildings_and_this_traj(tx, bs, buildings, rx, ax=self.list_figs['Scenario'][1][ri])

    def plot_buildings_and_this_traj(self, tx, bs, buildings, rx, ax=None, idx=None):
        if ax is not None:
            for line_i in buildings:
                ax.plot(line_i[0:2], line_i[2:4], color='black', linestyle='dashed')

            if idx is not None:
                ax.plot(tx[idx, 0], tx[idx, 1], ls='', color='red', marker='o', markersize=7)
                ax.plot(rx[bs]['x'], rx[bs]['y'], marker='x', color='red', markersize=20)
            else:
                ax.plot(tx[:, 0], tx[:, 1], color='red', marker='o', markersize=7)
                ax.plot(tx[0, 0], tx[0, 1], color='blue', marker='X', markersize=20)
               # ax.plot(tx[500, 0], tx[500, 1], color='green', marker='o', markersize=5)

                ax.plot(rx[bs]['x'], rx[bs]['y'], marker='X', color='red', markersize=20)

            ax.set_xticks([])
            ax.set_yticks([])
            #ax.set_xlabel('x [m]', fontsize=30)
            #ax.set_ylabel('y [m]', fontsize=30)

        else:
            plt.figure()
            for line_i in buildings:
                plt.plot(line_i[0:2], line_i[2:4], color='blue', linestyle='dashed')

            if idx is not None:
                plt.plot(tx[idx, 0], tx[idx, 1], color='red', marker='o', markersize=7)
                plt.plot(rx[bs]['x'], rx[bs]['y'], marker='X', color='red', markersize=20)
            else:
                plt.plot(tx[:, 0], tx[:, 1], color='red', marker='o', markersize=7)
                ax.plot(tx[0, 0], tx[0, 1], color='blue', marker='X', markersize=20)
                #ax.plot(tx[500, 0], tx[500, 1], color='green', marker='o', markersize=5)
                plt.plot(rx[bs]['x'], rx[bs]['y'], marker='X', color='red', markersize=20)

            plt.set_xticks([])
            plt.set_yticks([])
            #plt.set_xlabel('x [m]', fontsize=30)
            #plt.set_ylabel('y [m]', fontsize=30)

    def plot_stationary_regions(self, plt_idx_row, plt_idx_col, tx, bs, code_route, dir_route_sta_regions):
        """Plot stationary regions"""

        ri = self.right_idx_format(plt_idx_row, plt_idx_col)

        temp = re.compile("([a-zA-Z]+)([0-9]+)")
        fn = temp.match(code_route).groups()
        fn = fn[1]

        if bs == 'MKT':
            data = sio.loadmat(
                r'C:\Users\julia\OneDrive - UCL\Pièces jointes\MATLAB\VariablesToSave\Measurement Campaigns\Analysis measures\Stationary Regions\\' + dir_route_sta_regions + '\sta_reg_MKT_' + fn + '.mat')
        elif bs == 'MXW':
            data = sio.loadmat(
                r'C:\Users\julia\OneDrive - UCL\Pièces jointes\MATLAB\VariablesToSave\Measurement Campaigns\Analysis measures\Stationary Regions\\' + dir_route_sta_regions + '\sta_reg_MXW_' + fn + '.mat')

        color_reg = False

        data = data['regs']
        sz_regs = []
        for reg_i in data:
            reg2 = reg_i[0].flatten()

            if color_reg:
                self.list_figs['Stat-Regs'][1][ri].plot(tx[reg2 - 1, 0], tx[reg2 - 1, 1], color='blue', marker='o', markersize='3')
            else:
                self.list_figs['Stat-Regs'][1][ri].plot(tx[reg2 - 1, 0], tx[reg2 - 1, 1], color='red', marker='o', markersize='3')

            color_reg ^= True

            sz_regs.append(reg2.shape[0])

        print('Average size of sta regions [wavs]: ', np.mean(sz_regs) / ((3e8 / 3.8e9) * (30.2 / 1.5)))

        self.list_figs['Sh-Fit'][1][ri].set_xlabel('x [m]')
        self.list_figs['Sh-Fit'][1][ri].set_ylabel('y [m]')

