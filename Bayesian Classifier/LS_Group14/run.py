import matplotlib.pyplot as plt
import math
import numpy as np
from matplotlib.patches import Ellipse

#----------------------GLOBAL VARIABLES--------------------

idx = 1			#	1 -> Case1	2 -> Case2	3 -> Case3	4 -> Case 4

data_set_size, mean_dot_size, region_size = 6, 50, 0.05

#	indices ==> 0 -> Class1, 1 -> Class2, 2 -> Class3
total_classes = 3
X1 = []
X2 = []
Mu = []
g0x, g1x, g2x = [], [], []
Sigma, Sigma1, Sigma1inv = [], [], []
Sigma2, Sigma2inv = [], []
g0x, g1x, g2x = [], [], []
testX1 = []
testX2 = []
fig, ax = plt.subplots()

#---------------F U N C T I O N S--------------------

def __Cov(x1, mu1, x2, mu2):
	n, curr = len(x1), 0
	for i in range (0, n):
		curr += (float(x1[i]) - mu1) * (float(x2[i]) - mu2)
	curr /= n
	return curr

def __compare3(red, green, blue):
	if(red > green):				#	red > green
			if(red > blue):			#	red > blue
				return 0
			else:					# 	red < blue
				return 2
	else:							#	red < green
		if(green > blue):			# 	green > blue
			return 1
		else:						#	green < blue
			return 2

def __gx(x1, Mu1, x2, Mu2, sigma_sq):
	val = (float(x1) - float(Mu1)) ** 2 + (float(x2) - float(Mu2)) ** 2
	val = (-1 / (2.0 * sigma_sq) ) * val
	return val

def __gx2(x1, Mu1, x2, Mu2, Sigma1inv):
	k1, k2 = float(x1) - float(Mu1), float(x2) - float(Mu2)
	a, b, c, d = float(Sigma1inv[0][0]), float(Sigma1inv[0][1]), float(Sigma1inv[1][0]), float(Sigma1inv[1][1])
	val = k1*k1*(a) + k2*k2*(d) + k1*k2*(b + c)
	val = (-1 / 2.0) * val
	return val

def __gx34(x1, Mu1, x2, Mu2, Sigma2inv):
	x1, x2, Mu1, Mu2 = float(x1), float(x2), float(Mu1), float(Mu2)
	a, b, c ,d = float(Sigma2inv[0][0]), float(Sigma2inv[0][1]), float(Sigma2inv[1][0]), float(Sigma2inv[1][1])
	A = x1*x1*(a) + x2*x2*(d) + x1*x2*(b+c)
	B = Mu1*Mu1*(a) + Mu2*Mu2*(d) + Mu1*Mu2*(b+c)
	C = x1*Mu1*a + x1*Mu2*c + x2*Mu1*b + x2*Mu2*d
	mod = a * d - b * c
	val = (-1 / 2.0) * (A + B -2*C) - (1 / 2.0) * math.log(mod)
	return val

def __findRegion(idx):
	xmin, xmax, ymin, ymax = -10, 30, -20, 20
	ax.set_xlim([xmin, xmax])
	ax.set_ylim([ymin, ymax])
	x1 = xmin
	while(x1 <= xmax):
		x2 = ymin
		while(x2 <= ymax):
			temp = []
			for i in range (0, total_classes):
				if(idx == 0):
					val = __gx(x1, Mu[i][0], x2, Mu[i][1], sigma_sq)
				elif(idx == 1):
					val = __gx2(x1, Mu[i][0], x2, Mu[i][1], Sigma1inv)
				else:
					val = __gx34(x1, Mu[i][0], x2, Mu[i][1], Sigma2inv[i])
				temp.append(val)
			classNo = __compare3(temp[0], temp[1], temp[2])
			if(classNo == 0):
				g0x.append((x1, x2))
			elif(classNo == 1):
				g1x.append((x1, x2))
			else:
				g2x.append((x1, x2))
			x2 += 0.2
		x1 += 0.2


def __plotExtra(l1, l2):
	plt.title("Linearly separable data set")
	plt.xlabel(l1, fontsize=12)
	plt.ylabel(l2, fontsize=12)
	plt.tick_params(labelsize=10)
	plt.legend(('Class1','Class2','Class3'), loc='upper right', fontsize='small')

def __plot2(pltno, Xaxis, Yaxis, color, sz):
	plt.scatter(Xaxis, Yaxis, s = sz, color = color)
	__plotExtra('Feature1', 'Feature2')

def __plot(pltno, arr, color, sz):
	Xaxis, Yaxis = [], []
	for i in range (0, len(arr)):
		x1, x2 = arr[i]
		Xaxis.append(x1)
		Yaxis.append(x2)
	plt.scatter(Xaxis, Yaxis, s = sz, color = color)
	__plotExtra('Feature1', 'Feature2')

#--------Taking input and calulating the mean--------

for i in range (0,total_classes):
	if(i == 0):
		f = open('Class1.txt','r')
	elif(i == 1):
		f = open('Class2.txt','r')
	else:
		f = open('Class3.txt','r')
	lines = f.readlines()
	total = len(lines) * 0.75
	f.close()
	mu1, mu2 = 0, 0
	temp1, temp2, temp3, temp4, temp5 = [], [], [], [], []
	for i in range (0, len(lines)):
		x1, x2 = lines[i].split()
		if(i < total):
			mu1, mu2 = mu1 + float(x1), mu2 + float(x2)
			temp1.append(x1), temp2.append(x2)
		else:
			temp4.append(x1), temp5.append(x2)
	mu1, mu2 = mu1 / total, mu2 / total
	temp3.append(mu1), temp3.append(mu2)
	X1.append(temp1), X2.append(temp2)
	Mu.append(temp3)
	testX1.append(temp4), testX2.append(temp5)

#-----------------Calcuating Cov Matrix---------------------------

for i in range (0,total_classes):
	#	Class = i
	M11 = __Cov(X1[i], Mu[i][0], X1[i], Mu[i][0])
	M12 = __Cov(X1[i], Mu[i][0], X2[i], Mu[i][1])
	M21 = M12
	M22 = __Cov(X2[i], Mu[i][1], X2[i], Mu[i][1])
	row1, row2, covM = [], [], []
	row1.append(M11), row1.append(M12), row2.append(M21), row2.append(M22)
	covM.append(row1), covM.append(row2)
	Sigma.append(covM)

#------------------------------------------------------------------

idx -= 1

#-------------------------------------------------------------------------
#	A)		Covariance matrix for all the classes is the same and is sigma^2 I

if(idx == 0):
	sigma_sq = 0
	for i in range (0,total_classes):
		for j in range (0,2):
			for k in range (0,2):
				sigma_sq += Sigma[i][j][k]
	sigma_sq /= 12
	__findRegion(idx)


#-------------------------------------------------------------------------
#	B)		Full Covariance matrix for all the classes is the same and is Sigma.

elif(idx == 1):
	s11 = (Sigma[0][0][0] + Sigma[1][0][0] + Sigma[2][0][0]) / 3
	s12 = (Sigma[0][0][1] + Sigma[1][0][1] + Sigma[2][0][1]) / 3
	s21 = (Sigma[0][1][0] + Sigma[1][1][0] + Sigma[2][1][0]) / 3
	s22 = (Sigma[0][1][1] + Sigma[1][1][1] + Sigma[2][1][1]) / 3

	r1, r2 = [], []
	r1.append(s11), r1.append(s12)
	r2.append(s21), r2.append(s22)
	Sigma1.append(r1)
	Sigma1.append(r2)
	mod = s11 * s22 - s21 * s12
	r1, r2 = [], []
	r1.append(s22 / mod), r1.append(-s12 / mod)
	r2.append(-s21 / mod), r2.append(s11 / mod)
	Sigma1inv.append(r1), Sigma1inv.append(r2)

	__findRegion(idx)
# print Sigma1inv



#-------------------------------------------------------------------------
#	C)		Covariance matric is diagonal and is different for each class

elif(idx == 2):
	for i in range (0, total_classes):
		val = 0
		for j in range (0, 2):
			for k in range (0, 2):
				val += Sigma[i][j][k]
		val /= 4
		Sigma2.append([[val, 0],[0,val]])
	Sigma2inv = Sigma2


	__findRegion(idx)

#-------------------------------------------------------------------------
#	D)		 Full Covariance matrix for each class is different

else:
	for i in range (0, total_classes):
		a, b, c, d = float(Sigma[i][0][0]), float(Sigma[i][0][1]), float(Sigma[i][1][0]), float(Sigma[i][1][1])
		Sigma2.append([[a, b],[c, d]])
		mod = a*d - c*b
		a, b, c, d = a / mod, b / mod, c / mod, d / mod
		Sigma2inv.append([[d, -b],[-c, a]])

	__findRegion(idx)


#-------------------------------------------------------------------------
# Testing data
# Red - Class1, Green - Class2, Blue - Class3

c11, c12, c13, c21, c22, c23, c31, c32, c33 = 0, 0, 0, 0, 0, 0, 0, 0, 0
r1, r2, r3 = [], [], []
ConfM = []
for i in range (0, total_classes):
	for j in range (0, len(testX1[i])):
		x1, x2 = testX1[i][j], testX2[i][j]
		temp = []
		for k in range (0, total_classes):
			if(idx == 0):
				val = __gx(x1, Mu[k][0], x2, Mu[k][1], sigma_sq)
			elif(idx == 1):
				val = __gx2(x1, Mu[k][0], x2, Mu[k][1], Sigma1inv)
			else:
				val = __gx34(x1, Mu[k][0], x2, Mu[k][1], Sigma2inv[k])
			temp.append(val)
		classNo = __compare3(temp[0], temp[1], temp[2])
		if(classNo == 0):
			if(i == 0):
				c11 += 1
			elif(i == 1):
				c21 += 1
			else:
				c31 += 1
		elif(classNo == 1):
			if(i == 0):
				c12 += 1
			elif(i == 1):
				c22 += 1
			else:
				c32 += 1
		else:
			if(i == 0):
				c13 += 1
			elif(i == 1):
				c23 += 1
			else:
				c33 += 1

r1.append(c11), r1.append(c12), r1.append(c13)
r2.append(c21), r2.append(c22), r2.append(c23)
r3.append(c31), r3.append(c32), r3.append(c33)

ConfM.append(r1), ConfM.append(r2), ConfM.append(r3)

print ConfM[0]
print ConfM[1]
print ConfM[2]



#-------------------PLOTTING GRAPH HERE---------------------------

# plt.figure(1)

#	Region
__plot(111, g0x, '#ffaaca', data_set_size)			#	red
__plot(111, g1x, '#baffd4', data_set_size)			#	green
__plot(111, g2x, '#b7d3ff', data_set_size)			#	blue

#	Datapoints
__plot2(111, X1[0], X2[0], '#cc0036', data_set_size)	#	red
__plot2(111, X1[1], X2[1], '#006824', data_set_size)	#	green
__plot2(111, X1[2], X2[2], '#5100a5', data_set_size)	#	blue


plt.scatter(Mu[0][0], Mu[0][1], s = mean_dot_size, color='black', marker='*')
plt.scatter(Mu[1][0], Mu[1][1], s = mean_dot_size, color='black', marker='*')
plt.scatter(Mu[2][0], Mu[2][1], s = mean_dot_size, color='black', marker='*')

# __plot2(111, testX1[0], testX2[0], '#000000', data_set_size)

def contour_plot(m1, m2, ax):
	color = "black"
	minX,minY=float('inf'),float('inf')
	maxX,maxY=-float('inf'),-float('inf')
	x = np.zeros(len(m1))
	y = np.zeros(len(m2))
	for i in range(0, len(m1)):
		x[i] = m1[i]
		y[i] = m2[i]
	cov = np.cov(x, y)
	lambda_, v = np.linalg.eig(cov)
	lambda_ = np.sqrt(lambda_)
	for j in xrange(1, 6):
	    ell = Ellipse(xy=(np.mean(x), np.mean(y)),
	                  width=lambda_[0]*j, height=lambda_[1]*j,
	                  angle=np.rad2deg(np.arccos(v[0, 0])),
	                  edgecolor=color)
	    ell.set_facecolor('none')
	    ax.add_artist(ell)


#	Red

x, y = [], []
for i in range(0, len(X1[0])):
	x.append(X1[0][i])
	y.append(X2[0][i])
contour_plot(x, y, ax)


#	Green

x, y = [], []
for i in range(0, len(X1[1])):
	x.append(X1[1][i])
	y.append(X2[1][i])
contour_plot(x, y, ax)


#	Blue

x, y = [], []
for i in range(0, len(X1[2])):
	x.append(X1[2][i])
	y.append(X2[2][i])
contour_plot(x, y, ax)
	

plt.show()