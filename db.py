import os
import sqlite3
import pandas as pd


class Database:

    def __init__(self, database_directory, database_name, table_name, columns=None, drop_on_init=False):
        """
        database class
        param database_directory: simulation results directory
        param database_name: database name (filename)
        param table_name: table name
        param columns: column names for table
        param drop_on_init: boolean to drop database if exists
        """
        self.table_name = table_name
        self.database = f'{database_directory}{database_name}.db'

        if drop_on_init:
            self.drop()

        self.columns = columns
        self.conn = sqlite3.connect(self.database)
        self.c = self.conn.cursor()

        self.initialize_database()

    def initialize_database(self):
        self.c.execute(f"CREATE TABLE IF NOT EXISTS {self.table_name} ({self.columns})")
        print('...database ready\nusing %s table' % self.table_name)

    def insert(self, *args):
        """
        insert records into database
        param args: record values to insert into database
        """
        n_cols = len(self.columns.split(','))
        n_vars = len(args[0])
        if n_cols != n_vars:
            raise Exception('Database and variable size mismatch')
        mark = '?, ' * n_vars
        self.c.execute(f"INSERT INTO {self.table_name} ({self.columns}) VALUES ({mark[:-2]})", args[0])
        self.conn.commit()

    def query(self, column_name=None, item_value=None):
        """
        query database table and return result
        If column_name is None, read all results into pandas dataframe
        param column_name: column value to query
        param item_value: value to match
        return: pandas dataframe or record
        """
        if column_name is None:
            cmd = f"SELECT * FROM {self.table_name}"
            self.c.execute(cmd)
            return pd.read_sql_query(cmd, self.conn)
        else:
            cmd = f"SELECT * FROM {self.table_name} WHERE {column_name}={item_value}"
            self.c.execute(cmd)
            return self.c.fetchone()

    def drop(self):
        """
        drop filesystem database
        """
        try:
            os.remove(self.database)
            print(f'\n... {self.database} database dropped\n')
        except OSError:
            print('...database does not exist')

    def close(self):
        """
        close database connection
        """
        print('...closing database connection\n')
        self.conn.close()
