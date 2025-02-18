import psycopg2
from psycopg2 import Error
from sqlalchemy import create_engine

class ConnectionDB:
    def __init__(self):
        self.connection = None
        self.cursor = None
        self.engine = None

    def connect(self):
        try:
            self.connection = psycopg2.connect(
                host="localhost",
                database="supermarket_marketing",
                user="postgres",
                password="railan",
                port="5432"
            )
            self.cursor = self.connection.cursor()
            print("DB connection successful")
        except (Exception, Error) as error:
            print("Error while connecting to PostgreSQL", error)
    
    def connect_sqlalchemy(self):
        try:
            self.engine = create_engine('postgresql://postgres:railan@localhost:5432/supermarket_marketing')
            print("DB connection successful")
        except (Exception, Error) as error:
            print("Error while connecting to PostgreSQL", error)
    
    def close(self):
        if self.connection:
            self.cursor.close()
            self.connection.close()
            print("PostgreSQL connection is closed")
            
    def execute(self, query):
        try:
            self.cursor.execute(query)
            self.connection.commit()
            print("Query executed successfully")   
            # RETORNAR DADOS
            return self.cursor.fetchall() 
            
        except (Exception, Error) as error:
            self.connection.rollback()
            print("Error while executing query", error)
            
            
    