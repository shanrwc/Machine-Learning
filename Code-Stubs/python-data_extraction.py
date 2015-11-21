#!/usr/bin/python

##CSV--Comma (or other delimiter) Separated Values
import csv
#Reading
with open(<filename>,<file_options>) as f:
    reader = csv.reader(f, delimiter=',') #other possible options include: escapechar, quotechar, lineterminator, skipinitialwhitespace
    for row in reader:
        print row
#reader also has a next() function

#Writing
arr = [some array of text]
with open(<filename>,<file_options>) as f:
    writer = csv.writer(f)
    for ele in arr:
        writer.writerow(ele)
#writer also has a writerows() function

#################################################################################

##Excel spreadsheets
import xlrd

workbook = xlrd.open_workbook(datafile) #this function has many options to control the
#format read in and how the file is opened
sheet = workbook.sheet_by_index(0) #if your workbook has multiple sheets, specificy the number
row = sheet.row_values(<n>) #note that rows and cells are objects; you can get either the object or the value
cell = sheet.cell_value(<n>,<m>)
slice_of_columns = sheet.col_values(<col_number>,start_rowx=0,end_rowx=None) #the col_slice version will return the cells
slice_of_rows = sheet.row_values(<row_number>,start_colx=0,end_colx=None)
number_of_columns = sheet.ncols
number_of_rows = sheet.nrows

#Note: Excel dates are weird floating point numbers.  To convert them to something more readable:
xlrd.xldate_as_tuple(<excel time>,0) #this returns (year, month, day, hour, minute, second)
#Also note that Excel doesn't have a date data type, so cells won't know that they contain dates

#################################################################################################

#JSON
import json
import requests
import codecs

#to request information from website
params = {"api-key":<your key>}
r = requests.get(<some valid url query>, params = params)
if r.status_code == requests.codes.ok: result = r.json()
#or read from file
while open(filename,"r") as f:
    info = json.loads(f.read())

#Version I used for Ravelry API
import urllib2
import base64
request = urllib2.Request(<valid url request string>)
base64string = base64.encodestring('%s:%s' % (<access key>,<personal key>)).replace('\n','')
request.add_header("Authorization","Basic %s" %base64string)
response = urllib2.urlopen(request)
bookdata = json.load(response)

#At this point, the resulting object is effectively a dictionary

##############################################################################################

#XML

import xml.etree.ElementTree as ET

tree = ET.parse(filename) #xml file doesn't need to be opened first
#there is also a fromstring method to read in data
root = tree.getroot()
root.tag #will return a string
root.attrib #will return a dictionary of attributes, as pairs of strings
#you can dig deeper in the tree via index numbers--treat it like a list of lists, not a dictionary!

#findall('<tag>') finds all elements with this tag amongst the current element's children; find() finds the first tagged child.
#element.text returns the text content; element.get('<attribute>') returns the attribute's value
for country in root.findall('country'):
    rank = country.find('rank').text
    name = country.get('name')
    print name, rank
    # Singapore 4
    #Panama 68

#############################################################################

#HTML using Beautiful Soup
from bs4 import BeautifulSoup
soup = BeautifulSoup(html_doc,'html.parser') #lxml is another common parser
