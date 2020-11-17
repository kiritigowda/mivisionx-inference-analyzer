# Copyright (c) 2018 - 2020 Kiriti Nagesh Gowda, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import collections
import random
import os
import sys
import argparse
import csv
from datetime import date

__author__ = "Kiriti Nagesh Gowda"
__copyright__ = "Copyright 2018 - 2020, Kiriti Nagesh Gowda - mivisionx-inference-analyzer"
__license__ = "MIT"
__version__ = "0.9.0"
__maintainer__ = "Kiriti Nagesh Gowda"
__email__ = "Kiritigowda@gmail.com"
__status__ = "beta"

# import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--results_file_1', type=str, default='',
                    help='Inference Results File-1.csv')
parser.add_argument('--results_file_2', type=str, default='',
                    help='Inference Results File-2.csv')
parser.add_argument('--output_directory', type=str, default='',
                    help='Directory - directory to save results')
parser.add_argument('--output_filename', type=str, default='',
                    help='Results File prefix - results .csv file prefix')
parser.add_argument('--topK', type=int, default=1,
                    help='Results mismatch in Top K default: Top1 [Range: 1 - 5]')
args = parser.parse_args()

resultsFile1 = args.results_file_1
resultsFile2 = args.results_file_2
outputDirectory = args.output_directory
fileName = args.output_filename
topK = args.topK

if resultsFile1 == '' or outputDirectory == '' or resultsFile2 == '':
    print("ERROR - NO Arguments Passed, use --h option")
    exit()

if (not os.path.isfile(resultsFile1)):
    print("\nERROR: Inference Results File-1.csv missing")
    quit()
else:
    print("\nSUCCESS: Found Inference Results File-1.csv\n")

if (not os.path.isfile(resultsFile2)):
    print("\nERROR: Inference Results File-2.csv missing")
    quit()
else:
    print("\nSUCCESS: Found Inference Results File-1.csv\n")

if not os.path.exists(outputDirectory):
    os.makedirs(outputDirectory)

if fileName == '':
    fileName = "inferenceResultsDiff"

# validate arguments
if topK <= 0 or topK > 5:
    print("ERROR: Top K Results [type:INT range:1 to 5]")
    exit()


# TBD: Support other TopK results
if topK > 1:
    print("ERROR: Top K Results greater than 1 not supported in this Release")
    exit()

row_count = 0
row_count_1 = 0
row_count_2 = 0

with open(resultsFile1) as mode1:
    reader_1 = csv.reader(mode1)
    next(reader_1)
    data_1 = [r for r in reader_1]
    row_count_1 = len(data_1)

with open(resultsFile2) as mode2:
    reader_2 = csv.reader(mode2)
    next(reader_2)
    data_2 = [r for r in reader_2]
    row_count_2 = len(data_2)

if row_count_1 != row_count_2:
    print("ERROR: Number of entries in Inference Results are different")
    exit()
else:
    row_count = row_count_1

# help print
print("\nmivisionx-inference-analyzer - Inference Results Difference Checker V-"+__version__+"\n")

# date
today = date.today()
dateCreated = today.strftime("%b-%d-%Y")

# CLI Print
orig_stdout = sys.stdout

# HTML File
html_output_file = outputDirectory+'/'+fileName+'.html'
sys.stdout = open(html_output_file, 'w+')

# HTML Header
print"<?xml version=\"1.0\" encoding=\"UTF-8\" ?>"
print"<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0 Strict//EN\" \"http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd\">"
print"<html xmlns=\"http://www.w3.org/1999/xhtml\">"
print"\n"
print"<head>"
print"<meta http-equiv=\"Content-Type\" content=\"application/xml+xhtml; charset=UTF-8\" />"
print"<title>Inference Results Difference Checker</title>"
print"</head>"
print"\n"
print"<body style=\"color:white; background-color:black\">"
print"\n"
print"<pre>"
print("%-30s %-30s %-30s %-30s %-30s %-30s" % ("Image File Name", "Ground Truth Label", "Model A Top-1","Model B Top-1","Model A Prob-1","Model B Prob-1"))


for x in range(row_count):
    if topK == 1:
        if data_1[x][2] != data_2[x][2]: #or data_1[x][7] != data_2[x][7]
            print("%-30s %-30s %-30s %-30s %-30s %-30s" % (str(data_1[x][0]),str(data_1[x][1]),str(data_1[x][2]),str(data_2[x][2]),str(data_1[x][7]),str(data_2[x][7])))

print"\n"
print"</pre>"
print"</body>"
print"</html>"