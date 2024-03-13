<h1 style="color:#000;font-type:system-ui;">
Data management
</h1>

Common taks of the data management process include:

* Data cleaning
* Data storage
* Data transform

<h1 style="color:#000;font-type:system-ui;">
SQL
</h1>

## Common commands
!!! bug "Drop a table from a database"
    ```
    DROP TABLE table_name;
    ```

!!! bug "Create a table in a database"
    ```
    CREATE TABLE table_name(
        column1	integer PRIMARY KEY,
        column2		numeric(10,9)[6686]
    );
    ```

!!! bug "Count number of rows in a table"
    ```
    SELECT COUNT(*) AS count FROM table_name;
    ```

!!! bug "Select first x number of rows from a table"
    ```
    SELECT * FROM table_name LIMIT 10;
    ```

!!! bug "Select the i entry of the ARRAY-type column for all rows in a table"
    ```
    SELECT array_type_column_name[i] from table_name;
    ```

<h1 style="color:#000;font-type:system-ui;">
PySpark
</h1>

``SparkSession`` class

Is the entry point for working with structured data using Spark SQL.

``getOrCreate()`` method

Either returns an existing Spark session or creates a new one.

``SparkContext`` vs ``SparkSession`` classes

Lower-level control of functionality vs Higher-level control of functionality

<h1 style="color:#000;font-type:system-ui;">
IRFs storage in database
</h1>

There are multiple ways to store the Impulse Response Functions (IRFs) in database. 

Each file `IRF_MKT_#` or `IRF_MXW_#` contains the IRFs over a given number of time snapshots. With the interval between snapshots being around 33 ms. 

* One way is to store each time snapshot IRF as an ``ARRAY``, giving then a table with two columns: one for the time snapshot number (or id, equivalently) and other for the time snapshot IRF ``ARRAY``.

* Another way is to create a table with three columns:
    * One for the time snapshot number
    * Another for the delay tap number
    * Another for the IRF value

