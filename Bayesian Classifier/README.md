# PatternRecognition

This repo contains all the CS669 - Pattern Recognition assignments

## There are four cases on which the Bayes classifier is built

	(a) Covariance matrix for all the classes is the same and is σ^2 I
	(b) Full Covariance matrix for all the classes is the same and is Σ.
	(c) Covariance matric is diagonal and is different for each class
	(d) Full Covariance matrix for each class is different

## Dependencies
     Python 2.7
     Python matplotlib
     Python numpy

## To run any of the two datasets:
Dataset 1: 2-dimensional artificial data of 3 or 4 classes:

	(a) Linearly separable data set
	(b) Nonlinearly separable data set

Dataset 2: Real world data set

	1)   Open the run.py in that folder.
	2)   Change the line8 for Bayes classifier
          a)   idx = 1
          b)   idx = 2
          c)   idx = 3
          d)   idx = 4
    3)	 Run on the terminal:
    		$ python run.py

## The graphs are plotted for each cases of the Bayes Classifier

	1)	The photo is named as "XNM.png" where X the case number (A/ B/ C/ D) and N and M are the classes for which the graph has been ploted.
	2)	For the Confusion Matrix find the file "CM" in each folder.
	
## To calculate Precision, Recall, Accuracy and F Measure
     1)   Run on the terminal:
               $ g++ -std=c++14 calc.cpp
               # ./a.out
     2)   First line of the input contains the dimension of the Confusion Matrix
     3)   Then follows n lines containing n numbers denoting an element of the Matrix
     4)   Output:
               1st line: Precision
               2nd line: Recall
               3rd line: Accuracy
               4th line: F Measure
