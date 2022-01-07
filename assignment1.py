import pyspark
import findspark
findspark.init()
from pyspark import SparkContext

import wget

import sys
wget.download(sys.argv[1])
sc = SparkContext.getOrCreate()  

def isfloat(value):
    try:
        float(value)
        return True
    except:
        return False
    

def correctRows(p):
    if(len(p)==17):
        if(isfloat(p[5]) and isfloat(p[11])):
            if(float(p[5]) !=0 and float(p[11]) !=0):
                return p

#Getting the Data            
lines = sc.textFile(sys.srgv[2])

words = lines.map(lambda x: x.split(','))

#Data Cleaning
wordsCorr = words.filter(correctRows)
    
wordsAstuples = words.map(lambda x: (x[0], [x[1], 1]))
    
medDriver = wordsAstuples.map(lambda x: (x[0], x[1][1]))
    
counts = medDriver.reduceByKey(lambda x, y: x+y)
    
top10 = counts.top(10, lambda x: x[1])

dataToSave = sc.parallelize(top10).coalesce(1)
dataToSave.saveAsTextFile(sys.srgv[3])

#Task 2

def zero_check(x):
    if (x[1] > 0):
        return x

drive_mon = words.map(lambda x: (x[1], (float(x[14])) + float(x[16])))
drive_time = words.map(lambda x: (x[1], float(x[4])/60))
mon_total = drive_mon.reduceByKey(lambda x, y: x+y)
time_total = drive_time.reduceByKey(lambda x, y: x+y)

mon_total_cor = mon_total.filter(lambda x: zero_check(x))
time_total_cor = time_total.filter(lambda x: zero_check(x))

mon_time = mon_total_cor.join(time_total_cor)
average = mon_time.map(lambda x: (x[0], x[1][0]/x[1][1]))

top10_average = average.top(10, lambda x: x[1])

dataToSave1 = sc.parallelize(top10_average).coalesce(1)
dataToSave1.saveAsTextFile(sys.srgv[4])
