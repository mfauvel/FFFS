'''
Created on 11 juin 2014

@author: mfauvel
'''
import npfs as npfs
import scipy as sp

n = 300 # Number of samples
d = 200 # Number of dimension
noise = 0.2
sp.random.seed(1)
var= sp.random.random_integers(0,d-1,3)

x = sp.dot(sp.random.randn(n,d),sp.random.randn(d,d)) # Generate random samples
y = sp.ones((n,1))
t = 7*x[:,var[0]] -5.5*x[:,var[2]] + x[:,var[1]]**2
t += noise*sp.mean(t)*sp.random.randn(n) # add some noise

y[t>sp.mean(t)]=2

 
model = npfs.GMM()
model.learn_gmm(x, y)
yp = model.predict_gmm(x)[0]
yp.shape=y.shape
t = sp.where(yp==y)[0]
print float(t.size)/y.size

ids,error=model.forward_selection(x, y,delta=1.5,v=5)
print ids
print error
print var