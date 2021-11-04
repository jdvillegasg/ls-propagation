from LargeScaleProp import LargeScaleProp
from LargeScaleProp import LsPropPlots
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pylab import cm
from scipy import stats     # Gaussian distribution
import scipy.io as sio
import irfDataHandling as irfdh
import pandas as pd
from scipy import interpolate
import re

mpl.use('Qt4Agg')

# Set Plot configurations
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
    "font.size": 25,
    "axes.linewidth": 1.5
})

colors = cm.get_cmap('Set1', 10)

list_of_dists = ['alpha', 'beta', 'norm', 'expon',
                     'cauchy', 'gausshyper', 'gamma',
                     'laplace', 'nakagami', 'pareto', 'rayleigh',
                     'rice', 'uniform']

list_of_dists_reduced = ['beta', 'norm', 'expon','gamma',
                     'laplace', 'nakagami', 'rayleigh',
                     'rice', 'uniform']

base_dir = r"C:\Users\julia\OneDrive - UCL\Pièces jointes\MATLAB\VariablesToSave\Measurement Campaigns\Analysis measures\Received power"
lmd = 3e8/3.e9

bs = 'MXW'
routes_str = ['ParkingLot', 'MarieCurie', 'StBarbe']

subplt_var = {'MKT': {'rows': 3, 'cols': 3}, 'MXW': {'rows': 3, 'cols': 3}}
subplt_left_over = {'MKT': [], 'MXW': {'rows':[2], 'cols':[1, 2]}}

type_prop = {'MXW': {'StBarbe1': '', 'StBarbe2': '',
                     'MarieCurie6': '', 'MarieCurie7': '', 'MarieCurie8': '',
                     'ParkingLot9': '', 'ParkingLot10': ''},
             'MKT': {'StBarbe9': '',
                     'MarieCurie5': '', 'MarieCurie6': '', 'MarieCurie7': '', 'MarieCurie8': '',
                     'ParkingLot1': '', 'ParkingLot2': '', 'ParkingLot3': '', 'ParkingLot4': ''}}


# LoS
# MXW
#custom_sel_snaps = {'StBarbe1': np.arange(785, 1482),
#                    'StBarbe2': np.arange(503, 1079)}
#MKT
#custom_sel_snaps = {'MarieCurie5': np.arange(4550, 5752),
#                    'MarieCurie6': np.arange(4650, 5400),
#                    'MarieCurie7': np.arange(4750, 5543),
#                    'MarieCurie8': np.arange(4950, 5586)}

# NLoS
if bs == 'MKT':
    custom_sel_snaps = {'MarieCurie5': np.arange(1, 4550),
                        'MarieCurie6': np.arange(1, 4650),
                        'MarieCurie7': np.arange(1, 4750),
                        'MarieCurie8': np.arange(1, 4950)}
else:
    custom_sel_snaps = {'StBarbe1': np.r_[np.arange(1, 785), np.arange(1481, 7314)],
                        'StBarbe2': np.r_[np.arange(1, 503), np.arange(1078, 6686)]}

dir_route_sta_regions = 'UsingRMatrix' # 'THR_0_0_25' #'UsingRMatrix\CMD_0_7'
method_calc_sta_regions = 'fullrcorr' # 'cluspdp'
thr_sta_regions = 0.55 # Has different meaning depending the computation method used

#Control what to plot
activate_plots = {'RXP': True, # Rx Powers
                  'PL': True, # Path Loss
                  'Sh-Fit': True, # Gauss and BGF Sh KS fit
                  'Sh-QQ': True, # QQ plot Sh BGoF vs Sh Gauss
                  'KFactor-Route': False, # K Factor in trajectory and buildings
                  'CDF-Del-Spreads': False, # CDF Delay spreads KS BGoF
                  'CDF-Del-Avg':False, # CDF Delay avgs KS BGoF
                  'Del-Spreads-AllRoute': False, # Delay Spreads all meas this route vs no. sample
                  'Scenario': True, # Buildings and routes
                  'Stat-Regs': True # Stationary regions
                  }

plot_obj = LsPropPlots(activate_plots)

cnt_plot = 0
Pt_dBm = 23
sh_ks_bgof_distr = {}
ds_avg = {}

plt_idx_col = ''
plt_idx_row = ''

for route_name in routes_str:
    obj_handle = LargeScaleProp(base_dir, bs, route_name, lmd=lmd)
    file_number = obj_handle.routes_dict[bs][route_name]

    rxp = obj_handle.get_all_rx_power_this_route(bs, route_name, dB='on')
    rx = np.array([obj_handle.rx[bs]['x'], obj_handle.rx[bs]['y']])[np.newaxis]

    cnt_file_number = 0
    for a, rx_p in rxp.items():
        print(bs, route_name, a)

        plt_idx_row = np.unravel_index(cnt_plot, (subplt_var[bs]['rows'], subplt_var[bs]['cols']))[0]
        plt_idx_col = np.unravel_index(cnt_plot, (subplt_var[bs]['rows'], subplt_var[bs]['cols']))[1]

        file_path = obj_handle.utility_get_gps_file_path(bs, file_number, cnt_file_number)

        rx_p = np.asarray(rx_p)
        print(rx_p.shape)

        sample_number = np.linspace(start=1, stop=rx_p.shape[0], num=rx_p.shape[0])

        tx, info = obj_handle.get_coordinates_from_file(file_path, obj_handle.offset_scenario)
        tx = obj_handle.prune_trajectories(tx, bs, file_number[cnt_file_number])
        tx_interpolated = obj_handle.interpolate_coordinates(tx, rx_p.shape[0], 30.2)

        if type_prop[bs][a] != '':
            rxp_grouped_by_quantile, idx_group_members, quantiles_k_factor = obj_handle.get_snapshot_partition_LoS_NLoS_from_rxp(rx_p, dB='on')
            if activate_plots['KFactor-Route']:
                plot_obj.plot_k_factor_chosen_quantiles(plt_idx_row, plt_idx_col, tx_interpolated, idx_group_members[-1], bs)
            if type_prop[bs][a] == 'LoS':
                sel_snaps = idx_group_members[-1]
                sel_snaps.sort()
            elif type_prop[bs][a] == 'NLoS':
                sel_snaps = [j for i in idx_group_members[0:-1] for j in i]
                sel_snaps.sort()
            elif type_prop[bs][a] == 'OLoS':
                sel_snaps = idx_group_members[-2]
                sel_snaps.sort()
            elif type_prop[bs][a] == 'custom':
                sel_snaps = custom_sel_snaps[a]

            tx_interpolated = tx_interpolated[sel_snaps, :]
            rx_p = rx_p[sel_snaps]
            sample_number = sample_number[sel_snaps]
            print(tx_interpolated.shape)

        #reg_CI, regressor_CI, predictor_CI, sh_CI = obj_handle.empirical_distance_path_loss(rx_p, Pt_dBm, tx_interpolated, rx, False)
        reg_FI, regressor_FI, predictor_FI, sh_FI = obj_handle.empirical_distance_path_loss(rx_p, Pt_dBm, tx_interpolated, rx, True)

        sh = sh_FI

        if activate_plots['RXP']:
            plot_obj.plot_fig_rxpowers(plt_idx_row, plt_idx_col, sample_number, rx_p)
        if activate_plots['PL']:
            #plot_fig_pathloss(ax_pklot2, plt_idx_row, plt_idx_col, regressor_CI, predictor_CI, reg_CI, False)
            plot_obj.plot_fig_pathloss(plt_idx_row, plt_idx_col, regressor_FI, predictor_FI, reg_FI, True)
        if activate_plots['Sh-Fit']:
            mu_g_sh, sig_g_sh, ks_bgof_distr = plot_obj.plot_fig_distribution_fits(plt_idx_row, plt_idx_col, sh, 0.05, 'Shadowing [dB]', list_of_dists_reduced)
            sh_ks_bgof_distr[a] = ks_bgof_distr[0]
        if activate_plots['Sh-QQ']:
            plot_obj.plot_qq_fig(plt_idx_row, plt_idx_col, sh)
        if activate_plots['CDF-Del-Spreads']:
            m_ds, v_ds, ds_avg[a] = plot_obj.plot_delay_spreads(plt_idx_row, plt_idx_col, 0.05, cnt_file_number, bs, method_calc_sta_regions, thr_sta_regions, route_name, list_of_dists_reduced)

            if activate_plots['Del-Spreads-AllRoute']:
                ds_this_route = [k for ds in ds_avg.values() for k in ds.tolist()]
                plot_obj.plot_delay_spreads_route(ds_this_route)
        if activate_plots['CDF-Del-Avg']:
            plot_obj.plot_delay_avgs(plt_idx_row, plt_idx_col, 0.05, cnt_file_number, bs, method_calc_sta_regions, thr_sta_regions, route_name, list_of_dists_reduced)
        if activate_plots['Scenario']:
            plot_obj.plot_routes_and_buildings(plt_idx_row, plt_idx_col, tx_interpolated, bs, obj_handle.buildings, obj_handle.rx)
        if activate_plots['Stat-Regs']:
            plot_obj.plot_stationary_regions(plt_idx_row, plt_idx_col, tx_interpolated, bs, a, dir_route_sta_regions)

        cnt_file_number = cnt_file_number + 1
        cnt_plot = cnt_plot + 1

plt.show()

#file = irfdh.get_rx_power(r'C:\Users\julia\OneDrive - UCL\Pièces jointes\MATLAB\VariablesToSave\Measurement Campaigns\Analysis measures\IRFs\IRF_MXW_9.csv')
#print(file.head())