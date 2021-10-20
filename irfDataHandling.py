import pandas as pd


def get_rx_power(file_addr):
    """
    -Computes the received power from the measured IRF files .csv.
    -These IRF files are the MIMO-average of the absolute value squared of the actual IR tensor
    -The .csv format is:  rows--> delay bin,  columns--> time snapshot
    :param bs:
    :param file_addr:
    :return:
    """
    rxp_file=pd.read_csv(file_addr, header=None)


    return rxp_file

