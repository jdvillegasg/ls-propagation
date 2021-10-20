from LargeScaleProp import LargeScaleProp
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
activate_plot = [True,  # Rx Powers
                 True,  # Path Loss
                 True,  # Gauss and BGF Sh KS fit
                 True, # QQ plot Sh BGoF vs Sh Gauss
                 True, # K Factor in trajectory and buildings
                 False, # CDF Delay spreads KS BGoF
                 False, # CDF Delay avgs KS BGoF
                 False, # Delay Spreads all meas this route vs no. sample
                 False,  # Buildings and routes
                 True # Stationary regions
                 ]

test_obj = LargeScaleProp.LsPropPlots({'RXP': True,
                                       'PL': True,
                                       'Sh-Fit': True,
                                       'Sh-QQ': True,
                                       'KFactor-Route': True,
                                       'Stat-Regs': True})

print(test_obj.list_figs)

def right_idx_format(i1, i2):
    # Not allowed this option
    #if (i1 =='') & (i2 ==''):
    #    ri =

    if (i1 =='') & (i2 !=''):
        ri = i2
    if (i1 !='') & (i2 ==''):
        ri = i1
    if (i1 !='') & (i2 !=''):
        ri = (i1, i2)

    return ri

def plot_fig_rxpowers(ax_pklot1, plt_idx_row, plt_idx_col, sample_number, rx_p):
    ri = right_idx_format(plt_idx_row, plt_idx_col)

    ax_pklot1[ri].set_xlim(1, rx_p.shape[0])
    ax_pklot1[ri].set_ylim(np.min(rx_p), np.max(rx_p))

    n_x_ticks = 3
    dyn_range_x = sample_number.shape[0]
    ax_pklot1[ri].xaxis.set_major_locator(mpl.ticker.MultipleLocator(np.floor(dyn_range_x/n_x_ticks)))

    n_y_ticks = 3
    dyn_range_y = np.max(rx_p) - np.min(rx_p)

    N = 1000
    cumsum = np.cumsum(np.insert(rx_p, 0, 0))
    rxp_smooth = (cumsum[N:] - cumsum[:-N]) / float(N)
    #rxp_smooth = np.r_[rx_p[:N-1], rxp_smooth]
    rxp_smooth2 = np.r_[rxp_smooth, rx_p[-(N-1):]]

    px, py = obj_handle.segments_fit(np.arange(1, rxp_smooth.shape[0]+1), rxp_smooth, 12)
    ms = np.diff(py)/np.diff(px)

    tck = interpolate.splrep(px, py)
    d2 = interpolate.splev(np.arange(start=1, stop=rxp_smooth.shape[0]+1, step=1), tck, der=2)

    print('ROUGHNESS SPLINE: ', np.sum(np.square(d2)))
    print('STD DEV MEAN FCN: ', np.std(rxp_smooth))

    ax_pklot1[ri].yaxis.set_major_locator(mpl.ticker.MultipleLocator(np.floor(dyn_range_y/n_y_ticks)))
    ax_pklot1[ri].set_xlabel('Snapshot number', fontsize=30)
    ax_pklot1[ri].set_ylabel('$P_{RX}$ [dBm]', fontsize=30)

    ax_pklot1[ri].xaxis.set_tick_params(which='major', size=10, width=1.5, direction='in', top='on')
    ax_pklot1[ri].xaxis.set_tick_params(which='minor', size=10, width=1.5, direction='in', top='on')
    ax_pklot1[ri].yaxis.set_tick_params(which='major', size=10, width=1.5, direction='in', right='on')
    ax_pklot1[ri].yaxis.set_tick_params(which='minor', size=10, width=1.5, direction='in', right='on')

    ax_pklot1[ri].plot(sample_number, rx_p, ls='--', linewidth=1.2, alpha=0.7, color='b')
    #ax_pklot1[ri].plot(px, py, ls='-.', linewidth=1.5, color='r')
    ax_pklot1[ri].plot(np.arange(np.floor(N/2), np.floor(N/2) + rxp_smooth.shape[0]), rxp_smooth, ls='-.', linewidth=3, color='r')

def plot_fig_pathloss(ax_pklot2, plt_idx_row, plt_idx_col, regressor, predictor, reg, FI):
    ri = right_idx_format(plt_idx_row, plt_idx_col)

    d_tx_rx = pow(10, regressor / 10)

    #aux = np.arange(start=1, stop=np.max(d_tx_rx)+1, step=1)
    aux = np.arange(start=np.min(d_tx_rx), stop=np.max(d_tx_rx) + 1, step=1)
    reg_indp_var_interp = aux.reshape(aux.shape[0], 1)

    if FI:
        pred_L = reg.predict(10 * np.log10(reg_indp_var_interp))
        str_FI = 'FI'
    else:
        pred_L = reg.predict(10*np.log10(reg_indp_var_interp)) + 10*np.log10( np.square(4 * np.pi / obj_handle.lmd) )
        str_FI = 'CI'

    n_x_ticks = 3
    dyn_range_x = np.max(d_tx_rx) - 1
    #ax_pklot2[ri].xaxis.set_major_locator(mpl.ticker.MultipleLocator(np.floor(dyn_range_x / n_x_ticks)))

    n_y_ticks = 3
    dyn_range_y = np.abs(np.max(pred_L) - np.min(pred_L))
    #ax_pklot2[ri].yaxis.set_major_locator(mpl.ticker.MultipleLocator(np.floor(dyn_range_y/n_y_ticks)))

    ax_pklot2[ri].set_xlabel('Transceiver distance [m]', fontsize=30)
    ax_pklot2[ri].set_ylabel('L [dB]', fontsize=30)

    ax_pklot2[ri].xaxis.set_tick_params(which='major', size=10, width=1.5, direction='in', top='on')
    ax_pklot2[ri].xaxis.set_tick_params(which='minor', size=10, width=1.5, direction='in', top='on')
    ax_pklot2[ri].yaxis.set_tick_params(which='major', size=10, width=1.5, direction='in', right='on')
    ax_pklot2[ri].yaxis.set_tick_params(which='minor', size=10, width=1.5, direction='in', right='on')

    ax_pklot2[ri].plot(d_tx_rx, predictor, color='b', ls='', marker='x', markersize=5)
    ax_pklot2[ri].plot(reg_indp_var_interp, pred_L, color='k', linewidth=4)
    #ax_pklot2[ri].semilogx(d_tx_rx, predictor, color='b', ls='', marker='x', markersize=3)
    #ax_pklot2[ri].semilogx(reg_indp_var_interp, pred_L, color='k', linewidth=5)

    #ax_pklot2[ri].legend(labels=['Path Loss', 'n=' + str(np.around(reg.coef_[0], 2))],
    # bbox_to_anchor=(1, 0), loc=4, frameon=False, fontsize=16)

    ploss_exp = np.around(reg.coef_[0], 2)
    print('Path Loss Exponent' + str_FI + ': ', ploss_exp)

def plot_fig_distribution_fits(ax_pklot, plt_idx_row, plt_idx_col, signal, alpha, title):
    """Fit a gaussian distribution and BGoF to the shadowing signal and plot it"""
    ri = right_idx_format(plt_idx_row, plt_idx_col)
    print(ri)
    print(signal.shape)
    # Set up axes for plotting
    ax_pklot[ri].set_xlabel(title, fontsize=30)
    ax_pklot[ri].set_ylabel('Density', fontsize=30)

    ax_pklot[ri].xaxis.set_tick_params(which='major', size=10, width=1.5, direction='in', top='on')
    ax_pklot[ri].xaxis.set_tick_params(which='minor', size=10, width=1.5, direction='in', top='on')
    ax_pklot[ri].yaxis.set_tick_params(which='major', size=10, width=1.5, direction='in', right='on')
    ax_pklot[ri].yaxis.set_tick_params(which='minor', size=10, width=1.5, direction='in', right='on')

    # Fit the shadowing
    norm = getattr(stats, 'norm')
    mean_g, std_g = norm.fit(signal)
    parameters = norm.fit(signal)
    x = np.linspace(np.min(signal), np.max(signal), 1000)
    pdf_gaussian = norm.pdf(x, mean_g, std_g)

    # Kolmogorov-Smirnov test
    ks_test = stats.kstest(signal, 'norm', args=parameters)
    critical_value = stats.ksone.ppf(1-alpha/2, signal.shape[0])

    # BGF
    fitted_distr = obj_handle.best_gof_distr(signal, list_of_dists_reduced)
    best_gof_dist = getattr(stats, fitted_distr[0][0])
    pdf_best_gof = best_gof_dist.pdf(x, *fitted_distr[0][3])

    ax_pklot[ri].hist(signal, density=True)
    ax_pklot[ri].plot(x, pdf_gaussian, color='k', linewidth=1.5)
    ax_pklot[ri].plot(x, pdf_best_gof, color='r', linewidth=1.5)
    #ax_pklot[ri].legend(labels=['$\mathcal{N}$', 'BGF'], bbox_to_anchor=(0, 1), loc=2, frameon=False, fontsize=20)

    print('Sh. Gauss. KS fit statistic:', ks_test)
    print('Sh. Gauss. KS fit critical value:', critical_value)

    print('Sh. gauss mean: ', mean_g)
    print('Sh. gauss std: ', std_g)

    print(title + ' BGoF KS fit statistic:', fitted_distr[0][1])
    print(title + ' BGoF KS fit critical value:', critical_value)

    print(title + ' BGoF KS fit distr. name:', fitted_distr[0][0])
    print(title + ' BGoF KS fit distr. parameters:', fitted_distr[0][3])

    return mean_g, std_g, fitted_distr

def plot_qq_fig(ax_pklot, plt_idx_row, plt_idx_col, sh):
    ri = right_idx_format(plt_idx_row, plt_idx_col)

    ax_pklot[ri].xaxis.set_tick_params(which='major', size=10, width=1.5, direction='in',
                                                              top='on')
    ax_pklot[ri].xaxis.set_tick_params(which='minor', size=10, width=1.5, direction='in',
                                                              top='on')
    ax_pklot[ri].yaxis.set_tick_params(which='major', size=10, width=1.5, direction='in',
                                                              right='on')
    ax_pklot[ri].yaxis.set_tick_params(which='minor', size=10, width=1.5, direction='in',
                                                              right='on')

    stats.probplot(sh, dist='norm', plot=ax_pklot[ri])

def plot_k_factor_chosen_quantiles(ax_pklot, plt_idx_row, plt_idx_col, tx, idxs, bs):
    ri = right_idx_format(plt_idx_row, plt_idx_col)

    ax_pklot[ri].xaxis.set_major_locator(mpl.ticker.MultipleLocator(50))
    ax_pklot[ri].yaxis.set_major_locator(mpl.ticker.MultipleLocator(50))

    ax_pklot[ri].xaxis.set_tick_params(which='major', size=10, width=1.5, direction='in',
                                                              top='on')
    ax_pklot[ri].xaxis.set_tick_params(which='minor', size=10, width=1.5, direction='in',
                                                              top='on')
    ax_pklot[ri].yaxis.set_tick_params(which='minor', size=10, width=1.5, direction='in',
                                                              right='on')
    ax_pklot[ri].yaxis.set_tick_params(which='major', size=10, width=1.5, direction='in',
                                                              right='on')

    obj_handle.plot_buildings_and_this_traj(tx, bs, ax=ax_pklot[plt_idx_row][plt_idx_col], idx=idxs)

def plot_delay_spreads(ax_pklot, plt_idx_row, plt_idx_col, alpha, cnt, bs, method, thr_level):
    """Find out the BGoF distribution for the delay spreads"""
    ri = right_idx_format(plt_idx_row, plt_idx_col)

    ax_pklot[ri].xaxis.set_tick_params(which='major', size=10, width=1.5, direction='in', top='on')
    ax_pklot[ri].xaxis.set_tick_params(which='minor', size=10, width=1.5, direction='in', top='on')
    ax_pklot[ri].yaxis.set_tick_params(which='minor', size=10, width=1.5, direction='in', right='on')
    ax_pklot[ri].yaxis.set_tick_params(which='major', size=10, width=1.5, direction='in', right='on')

    dict_var_delay = {'MXW': {'StBarbe': [0, 1], 'MarieCurie': [2, 3, 4], 'ParkingLot': [5, 6]},
                      'MKT': {'ParkingLot': [0, 1, 2, 3], 'MarieCurie': [4, 5, 6, 7], 'StBarbe': [8]}}

    if bs=='MKT':
        data = sio.loadmat(r'C:\Users\julia\OneDrive - UCL\Pièces jointes\MATLAB\var_delay_statistics_sta_regs_' + method + '_thr_' + str(thr_level) + '_MKT.mat')
    elif bs=='MXW':
        data = sio.loadmat(r'C:\Users\julia\OneDrive - UCL\Pièces jointes\MATLAB\var_delay_statistics_sta_regs_' + method + '_thr_' + str(thr_level) + '_MXW.mat')

    ds = data['delay_spreads'][dict_var_delay[bs][route_name]]

    ds_avg = np.mean(ds[cnt][0], axis=1)*1e9
    critical_value = stats.ksone.ppf(1 - alpha / 2, ds_avg.shape[0])
    fitted_distr_ds = obj_handle.best_gof_distr(ds_avg, list_of_dists_reduced)

    x_ds = np.linspace(np.min(ds_avg), np.max(ds_avg), 1000)
    best_gof_dist_ds = getattr(stats, fitted_distr_ds[0][0])
    cdf_best_gof_ds = best_gof_dist_ds.cdf(x_ds, *fitted_distr_ds[0][3])

    ax_pklot[ri].hist(ds_avg, 50, density=True, cumulative=True, color='b', histtype='step')
    #ax_pklot[ri].hist(ds_avg, 50, density=True, color='b')
    ax_pklot[ri].plot(x_ds, cdf_best_gof_ds, color='b', linewidth=1.5, ls='--', label=str(cnt))
    ax_pklot[ri].set_xlabel('Delay spreads [$\mu$s]')
    ax_pklot[ri].set_ylabel('CDF')
    ax_pklot[ri].legend(labels=['Fit: ' + fitted_distr_ds[0][0], 'Empirical'],
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

def plot_delay_avgs(ax_pklot, plt_idx_row, plt_idx_col, alpha, cnt, bs, method, thr_level):
    """Find out the BGoF distribution for the delay avgs"""
    ri = right_idx_format(plt_idx_row, plt_idx_col)

    ax_pklot[ri].xaxis.set_tick_params(which='major', size=10, width=1.5, direction='in', top='on')
    ax_pklot[ri].xaxis.set_tick_params(which='minor', size=10, width=1.5, direction='in', top='on')
    ax_pklot[ri].yaxis.set_tick_params(which='minor', size=10, width=1.5, direction='in', right='on')
    ax_pklot[ri].yaxis.set_tick_params(which='major', size=10, width=1.5, direction='in', right='on')

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
    fitted_distr_dm = obj_handle.best_gof_distr(dm_avg, list_of_dists_reduced)

    x_dm = np.linspace(np.min(dm_avg), np.max(dm_avg), 1000)
    best_gof_dist_dm = getattr(stats, fitted_distr_dm[0][0])
    cdf_best_gof_dm = best_gof_dist_dm.cdf(x_dm, *fitted_distr_dm[0][3])

    ax_pklot[ri].hist(dm_avg, 50, density=True, cumulative=True, color='b', histtype='step')
    #ax_pklot[ri].hist(dm_avg, 50, density=True, color='b')
    ax_pklot[ri].plot(x_dm, cdf_best_gof_dm, color='b', linewidth=1.5, ls='--', label=str(cnt))
    ax_pklot[ri].set_xlabel('Delay averages [$\mu$s]')
    ax_pklot[ri].set_ylabel('PDF')
    ax_pklot[ri].legend(labels=['Fit: ' + fitted_distr_dm[0][0], 'Empirical'],
                        bbox_to_anchor=(0, 1), loc=2, frameon=False, fontsize=16)

    median_dm = np.quantile(dm_avg, 0.5)
    std_dm = np.std(dm_avg)

    print('Median Del. Avgs Route ' + route_name + str(cnt) + ' : ' + str(median_dm))
    print('Std Dev Del. Avgs Route ' + route_name + str(cnt) + ' : ' + str(std_dm))

    print('Del. Avgs KS fit statistic:', fitted_distr_dm[0][1])
    print('Del. Avgs KS fit critical value:', critical_value)

    print('Del. Avgs BGoF disitr. name: ', fitted_distr_dm[0][0])
    print('Del. Avgs KS fit distr. parameters:', fitted_distr_dm[0][3])

def plot_delay_spreads_route(ax_pklot, ds):

    ax_pklot.xaxis.set_tick_params(which='major', size=10, width=1.5, direction='in',
                                                             top='on')
    ax_pklot.xaxis.set_tick_params(which='minor', size=10, width=1.5, direction='in',
                                                             top='on')
    ax_pklot.yaxis.set_tick_params(which='minor', size=10, width=1.5, direction='in',
                                                             right='on')
    ax_pklot.yaxis.set_tick_params(which='major', size=10, width=1.5, direction='in',
                                                             right='on')

    ax_pklot.plot(ds, marker='x', markersize=5)

def plot_routes_and_buildings(ax_pklot, plt_idx_row, plt_idx_col, tx, bs):
    """Wrapper to plot_buildings_and_this_traj"""
    ri = right_idx_format(plt_idx_row, plt_idx_col)
    obj_handle.plot_buildings_and_this_traj(tx, bs, ax=ax_pklot[ri])

def plot_stationary_regions(ax_pklot, plt_idx_row, plt_idx_col, tx, bs):
    """Plot stationary regions"""

    ri = right_idx_format(plt_idx_row, plt_idx_col)

    temp = re.compile("([a-zA-Z]+)([0-9]+)")
    fn = temp.match(a).groups()
    fn = fn[1]

    if bs == 'MKT':
        data = sio.loadmat(r'C:\Users\julia\OneDrive - UCL\Pièces jointes\MATLAB\VariablesToSave\Measurement Campaigns\Analysis measures\Stationary Regions\\' + dir_route_sta_regions + '\sta_reg_MKT_' + fn + '.mat')
    elif bs == 'MXW':
        data = sio.loadmat(r'C:\Users\julia\OneDrive - UCL\Pièces jointes\MATLAB\VariablesToSave\Measurement Campaigns\Analysis measures\Stationary Regions\\' + dir_route_sta_regions + '\sta_reg_MXW_'  + fn + '.mat')

    color_reg = False

    data = data['regs']
    sz_regs = []
    for reg_i in data:
        reg2 = reg_i[0].flatten()

        if color_reg:
            ax_pklot[ri].plot(tx[reg2-1, 0], tx[reg2-1, 1], color='blue', marker='o', markersize='3')
        else:
            ax_pklot[ri].plot(tx[reg2-1, 0], tx[reg2-1, 1], color='red', marker='o', markersize='3')

        color_reg ^= True

        sz_regs.append(reg2.shape[0])

    print('Average size of sta regions [wavs]: ', np.mean(sz_regs)/((3e8/3.8e9)*(30.2/1.5)))

    ax_pklot[ri].set_xlabel('x [m]')
    ax_pklot[ri].set_ylabel('y [m]')

cnt_plot = 0
Pt_dBm = 23
sh_ks_bgof_distr = {}
ds_avg = {}

plt_idx_col = ''
plt_idx_row = ''

if activate_plot[0]:
    fig1, ax_pklot1 = plt.subplots(subplt_var[bs]['rows'], subplt_var[bs]['cols'])
    if subplt_left_over[bs]:
        for i in subplt_left_over[bs]['rows']:
            for j in subplt_left_over[bs]['cols']:
                ax_pklot1[i, j].set_visible(False)
if activate_plot[1]:
    fig2, ax_pklot2 = plt.subplots(subplt_var[bs]['rows'], subplt_var[bs]['cols'])
    if subplt_left_over[bs]:
        for i in subplt_left_over[bs]['rows']:
            for j in subplt_left_over[bs]['cols']:
                ax_pklot2[i, j].set_visible(False)
if activate_plot[2]:
    fig3, ax_pklot3 = plt.subplots(subplt_var[bs]['rows'], subplt_var[bs]['cols'])
    if subplt_left_over[bs]:
        for i in subplt_left_over[bs]['rows']:
            for j in subplt_left_over[bs]['cols']:
                ax_pklot3[i, j].set_visible(False)
if activate_plot[3]:
    fig5, ax_pklot4 = plt.subplots(subplt_var[bs]['rows'], subplt_var[bs]['cols'])
    if subplt_left_over[bs]:
        for i in subplt_left_over[bs]['rows']:
            for j in subplt_left_over[bs]['cols']:
                ax_pklot4[i, j].set_visible(False)
if activate_plot[4]:
    fig6, ax_pklot5 = plt.subplots(subplt_var[bs]['rows'], subplt_var[bs]['cols'])
    if subplt_left_over[bs]:
        for i in subplt_left_over[bs]['rows']:
            for j in subplt_left_over[bs]['cols']:
                ax_pklot5[i, j].set_visible(False)
if activate_plot[5]:
    fig7, ax_pklot6 = plt.subplots(subplt_var[bs]['rows'], subplt_var[bs]['cols'])
    if subplt_left_over[bs]:
        for i in subplt_left_over[bs]['rows']:
            for j in subplt_left_over[bs]['cols']:
                ax_pklot6[i, j].set_visible(False)
if activate_plot[6]:
    fig7, ax_pklot7 = plt.subplots(subplt_var[bs]['rows'], subplt_var[bs]['cols'])
    if subplt_left_over[bs]:
        for i in subplt_left_over[bs]['rows']:
            for j in subplt_left_over[bs]['cols']:
                ax_pklot7[i, j].set_visible(False)
if activate_plot[7]:
    fig8, ax_pklot8 = plt.subplots(1, 1)
if activate_plot[8]:
    fig9, ax_pklot9 = plt.subplots(subplt_var[bs]['rows'], subplt_var[bs]['cols'])
    if subplt_left_over[bs]:
        for i in subplt_left_over[bs]['rows']:
            for j in subplt_left_over[bs]['cols']:
                ax_pklot9[i, j].set_visible(False)
if activate_plot[9]:
    fig9, ax_pklot10 = plt.subplots(subplt_var[bs]['rows'], subplt_var[bs]['cols'])
    if subplt_left_over[bs]:
        for i in subplt_left_over[bs]['rows']:
            for j in subplt_left_over[bs]['cols']:
                ax_pklot10[i, j].set_visible(False)


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
            if activate_plot[4]:
                plot_k_factor_chosen_quantiles(ax_pklot5, plt_idx_row, plt_idx_col, tx_interpolated,
                                               idx_group_members[-1],
                                               bs)
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

        if activate_plot[0]:
            plot_fig_rxpowers(ax_pklot1, plt_idx_row, plt_idx_col, sample_number, rx_p)
        if activate_plot[1]:
            #plot_fig_pathloss(ax_pklot2, plt_idx_row, plt_idx_col, regressor_CI, predictor_CI, reg_CI, False)
            plot_fig_pathloss(ax_pklot2, plt_idx_row, plt_idx_col, regressor_FI, predictor_FI, reg_FI, True)
        if activate_plot[2]:
            mu_g_sh, sig_g_sh, ks_bgof_distr = plot_fig_distribution_fits(ax_pklot3, plt_idx_row, plt_idx_col, sh, 0.05,
                                                               'Shadowing [dB]')
            sh_ks_bgof_distr[a] = ks_bgof_distr[0]
        if activate_plot[3]:
            plot_qq_fig(ax_pklot4, plt_idx_row, plt_idx_col, sh)
        if activate_plot[5]:
            m_ds, v_ds, ds_avg[a] = plot_delay_spreads(ax_pklot6, plt_idx_row, plt_idx_col, 0.05, cnt_file_number, bs, method_calc_sta_regions, thr_sta_regions)

            if activate_plot[7]:
                ds_this_route = [k for ds in ds_avg.values() for k in ds.tolist()]
                plot_delay_spreads_route(ax_pklot8, ds_this_route)
        if activate_plot[6]:
            plot_delay_avgs(ax_pklot7, plt_idx_row, plt_idx_col, 0.05, cnt_file_number, bs, method_calc_sta_regions, thr_sta_regions)
        if activate_plot[8]:
            plot_routes_and_buildings(ax_pklot9, plt_idx_row, plt_idx_col, tx_interpolated, bs)
        if activate_plot[9]:
            plot_stationary_regions(ax_pklot10, plt_idx_row, plt_idx_col, tx_interpolated, bs)

        cnt_file_number = cnt_file_number + 1
        cnt_plot = cnt_plot + 1

lines = []
labels = []

# for ax in fig3.axes:
#     axLine, axLabel = ax.get_legend_handles_labels()
#     lines.extend(axLine)
#     labels.extend(axLabel)
#
# fig3.legend(lines, labels=['$\mathcal{N}$', 'BGF'], loc='upper right', mode="expand", ncol=3, fontsize=20)

plt.show()

file = irfdh.get_rx_power(r'C:\Users\julia\OneDrive - UCL\Pièces jointes\MATLAB\VariablesToSave\Measurement Campaigns\Analysis measures\IRFs\IRF_MXW_9.csv')
print(file.head())