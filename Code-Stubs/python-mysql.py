#!/usr/bin/python

import MySQLdb
import MySQLdb.cursors
import time #if you're going to be working with time objects
import datetime

##Create database connection
conn = MySQLdb.connect(host="localhost",
                       user="guest",
                       passwd="",
                       db=<database>,
                       cursorclass=MySQLdb.cursors.DictCursor)
#Include last argument if you want queries to return dictionaries,
#instead of the default lists

x = conn.cursor()
x.execute("""<some valid sql query %s""" % (str(something)))
conn.commit()
#Use after adding to database

#If your query returned information
my_info = x.fetchall()
#The above can be a list of lists, or a list of dictionaries depending on your cursor

for row in my_info:
    for col in row: print col

conn.close()
