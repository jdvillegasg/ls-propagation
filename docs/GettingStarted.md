# Spark

* Install JDK 

Download and install the JDK version of your choice from Oracle's website.

* Install Apache Spark

Download and install Apache Spark version from Apache's website.

* Install PySpark library 

```bash
pip install pyspark
```

* Install Findspark library

```bash
pip install findspark
```

* Set ``JAVA_HOME`` and ``SPARK_HOME`` environment variables on Windows

* Add a ``PYSPARK_DRIVER_PYTHON`` environment variable and set its value to ``jupyter``.

* Add a ``PYSPARK_DRIVER_PYTHON_OPTS`` environment variable and set its value to ``notebook``

* Add ``JAVA_HOME`` to the ``PATH`` system variable in Windows, for it to be used by any user.

* From the downloaded Spark filename check the Hadoop version (mine was 3), and download the ``winutils.exe`` (corresponding to that Hadoop version) file from [THIS](https://github.com/steveloughran/winutils) repository.

* Place the ``winutils.exe`` file in the ``[SPARK_HOME]\bin`` directory.

* Restart your local Windows machine for some of these changes to take effect (in case there is any problem).

# PostgreSQL

* When installing PostgreSQL one must provide a password for the superuser (admin).

Password: `postgrejulian`

* The port where the server listen is set to the default `5432`.

* Set as `Default locale` the locale of the new database cluster.

* Password for the user `postgres` to connect to the server `PostgreSQL 16` is `postgrejulian`


 cursor.execute("""
                INSERT INTO """ +  table_name + """ ("""+column_names[0]+""", """+column_names[1]+""")
                SELECT col1, col2 FROM temp_table
            """)