import os
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values
import csv
import logging
from constants.db_info import DB_HOST, DB_USER, DB_NAME, DB_PASSWORD, DB_PORT
import numpy as np
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataAdmin(object):
    def __init__(self):
        """
        The properties of the class are:
        * ``precision``: defining the total number of digits of a numeric value
        * ``scale``: defining the decimal digits of a numeric value
        """
        self.precision = 20
        self.scale = 19

    def insert_from_csv_to_tb(self, csv_file_path, table_name):
        """
        Creates and populates a table from a csv file.
        
        The csv files are structured as follows:
        * The columns of each file represent the Power Delay Profile (PDP).
        * The number of colums refer to the number of Time snapshots a given measurement had.
        * All the files have the same number of rows. The number of rows (2044) represent the delay taps of the PDP.
        
        Args:
            csv_file_path (str): path to csv file.
            table_name (str): name of the table to be created in the database
        """
        
        try:
            conn = psycopg2.connect(
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
                host=DB_HOST,
                port=DB_PORT
            )    
        except psycopg2.Error as e:
            logger.error(f"Error connecting to database: {e}")
                   
        IRF = []
        with open(csv_file_path,'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                IRF.append([float(value) for value in row])
        
        data_to_insert = [(snap_id, delay_id, irf_value) 
                            for delay_id, row in enumerate(IRF, start=1) 
                            for snap_id, irf_value in enumerate(row, start=1)]

        del IRF
        
        try:
            cur = conn.cursor()

            cur.execute("""
                CREATE TABLE IF NOT EXISTS """+table_name+""" (
                    snap_id integer,
                    delay_id INTEGER,
                    irf_value NUMERIC("""+str(self.precision)+""","""+str(self.scale)+""")
                );
            """)            

            execute_values(cur, """
                INSERT INTO """+table_name+""" (snap_id, delay_id, irf_value) 
                VALUES %s;
            """, data_to_insert)
            
            conn.commit()

            # Check if data has been inserted
            cur.execute("SELECT COUNT(*) FROM " + table_name)
            count = cur.fetchone()[0]
            logger.info(f"Total rows in table {table_name} after insertion: {count}")

        except psycopg2.Error as e:
            logger.error(f"Error performing queries on {table_name}: {e}")

        finally:
            # Commit the transaction and close the cursor and connection
            if cur:
                cur.close()
            if conn:    
                conn.close()
    
    def get_rx_power(self, table_name, scale='linear'):
        """
        Compute the received power in dB from a table in a database.

        Args:
            table_name (str): table name in database.
            array_type_column_name (str): column name having the array type data.
            nsnaps (int): number of time snapshots. Number of rows in the table.
            scale (str): 'dB' for computing the power in dB. Otherwise, linear scale.

        Returns:
            rx_power (list): computed received power.
        """
        try:
            conn = psycopg2.connect(
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
                host=DB_HOST,
                port=DB_PORT
            )
        except psycopg2.Error as e:
            logger.error(f"Error connecting to database: {e}")
            rx_power = -1
        
        try:
            cursor = conn.cursor()
        
            cursor.execute("""
                SELECT snap_id,
                SUM(irf_value) AS total_sum
                FROM """+table_name+"""
                WHERE irf_value <> 'NaN'
                GROUP BY snap_id
                ORDER BY snap_id;           
            """
            )
        except psycopg2.Error as e:
            logger.error(f"Error when querying table {table_name}: {e}")
            rx_power = -1            
        else:
            rx_power = cursor.fetchall()
            rx_power = [i[1] for i in rx_power]
                
            if cursor:
                cursor.close()
            if conn:    
                conn.close()
                
        return rx_power

    def insert_all_csv_to_tb(self, csv_folder, table_names=None, except_files=None):
        """
        Creates and populates multiple tables from a folder having csv files.

        Args:
            csv_folder (str): path to the folder containing the csv files. Must have solely csv files
            column_names (list): list of strings with the names of the columns for each table.
            table_names (str, optional): the name to give to the table. If not provided, the name of the table will be the name of the csv file. Defaults to None.
            except_files(list, optional): list of strings having the filenames which we have already created and populated.
        """
        list_filenames = os.listdir(csv_folder)
        
        for cnt, filename in enumerate(list_filenames):
            if except_files is not None:
                if filename.split('.')[0] in except_files: 
                    continue
            if table_names is None:
                tb_name = filename.split('.')[0]
            else:
                tb_name = table_names[cnt]
                
            self.insert_from_csv_to_tb(f"{csv_folder}\{filename}", tb_name)
    
    def get_pdp(self, table_name, t_snap):
        """
        Queries the *Power Delay Profile (PDP)* for a given time snapshot and table name.
        
        Args:
            table_name (str): table name in database.
            t_snap (int): time snapshot number.
        Returns:
            response_to_query (list): Power Delay Profile for a given time snapshot.
        """        
        
        try:
            conn = psycopg2.connect(
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
                host=DB_HOST,
                port=DB_PORT
            )
        except psycopg2.Error as e:
            logger.error(f"Error connecting to the database: {e}")
        
        try: 
            cur = conn.cursor()
            
            cur.execute(
                """
                SELECT irf_value FROM """+table_name+"""
                WHERE snap_id = """+str(t_snap)+"""
                ORDER BY delay_id;
                """
            )
        
            response_to_query = cur.fetchall()
            response_to_query = [i[0] for i in response_to_query]
        except psycopg2.Error as e:
            logger.error(f"Error when executing the query: {e}")
        else:
            if cur:
                cur.close()
            if conn:    
                conn.close()
            return response_to_query