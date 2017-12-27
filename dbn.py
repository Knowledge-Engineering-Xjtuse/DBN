import numpy as np
import idx2numpy
import Image
import pickle
import matplotlib.pyplot as plt

images=idx2numpy.convert_from_file("train-images-idx3-ubyte")
data=[]
temp=[]
for image in images:
    for i in image:
        for j in i:
            temp.append(j)
    data.append(temp)
    temp=[]
X = np.asarray(data, 'float32')
X = X/255.0
Y = idx2numpy.convert_from_file("train-labels-idx1-ubyte")
X_test = X[50000:]
Y_test = Y[50000:]
X = X[:50000]
Y = Y[:50000]
sizes=[X.shape[1],100]

class rbm:
    def __init__(self,sizes=[],learning_rate=0.01,numepochs = 1):
        print 'rbm init ,sizes:',sizes,', numepochs:',numepochs
        self.W=np.matrix(np.zeros(sizes,'float32'))
        self.vW=np.matrix(np.zeros(sizes,'float32'))

        self.b=np.matrix(np.zeros(sizes[0],'float32'))
        self.vb=np.matrix(np.zeros(sizes[0],'float32'))

        self.c=np.matrix(np.zeros(sizes[1],'float32'))
        self.vc=np.matrix(np.zeros(sizes[1],'float32'))

        self.learning_rate = learning_rate
        self.numepochs = numepochs

        self.v1=self.b
        self.v2=self.b
        self.h1=self.c
        self.h2=self.c
        self.c1=self.W
        self.c2=self.W

    def sigmrnd(self,X):
        print np.float32(1./(1.+np.exp(-X)) > np.random.random(X.shape))
        return np.float32(1./(1.+np.exp(-X)) > np.random.random(X.shape))

    def train(self,X):
        print 'begin train , X.shape',X.shape,'numepochs:',self.numepochs
        for i in range(self.numepochs):
            err = 0
            index = []
            k = 0
            reX = []
            reY = []
            for v in X:
                self.v1 = np.matrix(v)

                self.h1 = self.sigmrnd(self.c+np.dot(self.v1,self.W))
                self.v2 = self.sigmrnd(self.b+np.dot(self.h1,self.W.T))
                self.h2 = self.sigmrnd(self.c+np.dot(self.v2,self.W))


                self.c1 = np.dot(self.h1.T,self.v1).T
                self.c2 = np.dot(self.h2.T,self.v2).T

                self.vW = self.learning_rate*(self.c1-self.c2)
                self.vb = self.learning_rate*(self.v1-self.v2)
                self.vc = self.learning_rate*(self.h1-self.h2)

                self.W = self.W + self.vW
                self.b = self.b + self.vb
                self.c = self.c + self.vc

                err = err + np.linalg.norm(self.v1-self.v2)

                index.append(k)
                reX.append(np.linalg.norm(self.v1))
                reY.append(np.linalg.norm(self.v2))
                k = k + 1
            print "err :", err
            # if i == self.numepochs - 1:
            #     plt.plot(index, reX, 'r')
            #     plt.plot(index, reY, 'g')
            #     plt.show()
        return self.W,self.b,self.c
    def predict(self,X):
        m,n = X.shape
        return self.sigmrnd(np.dot(np.ones((m,1)),self.c)+np.dot(X,self.W))

# rbm1=rbm(sizes)
# rbm1.train(X)
#print rbm1.predict(X)

#picklestring = pickle.dumps(rbm1)
#f=open('rbm1','w')
#f.write(picklestring)

#f=open('rbm1','r')
#rbm1=pickle.load(f)
#f.close()

class dbn:
    def __init__(self,sizes = [],learning_rate = 0.01,numepochs = 1):

        print 'dbn init ,sizes:',sizes,', numepochs:',numepochs
        self.sizes = sizes
        self.rbms = []
        self.learning_rate = learning_rate
        self.numepochs = numepochs
        self.X = []

        for i in range(len(self.sizes)-1):
            self.rbms.append(rbm(sizes[i:i+2],0.01,self.numepochs))

    def train(self,X):
        self.X.append(X)
        for j in range(len(self.sizes)-1):
            self.rbms[j].train(self.X[j])
            self.X.append(self.rbms[j].predict(self.X[j]))

    def predict(self,X):
        for j in range(len(self.sizes)-1):
            X = self.rbms[j].predict(X)
        return X

sizes = [X.shape[1],200,100]
dbn1 = dbn(sizes,0.01,4)
dbn1.train(X)

def sigmoid(X):
     return 1.0/(1.0+np.exp(-X))

def gradAscent(data,labels,label):
    m,n=data.shape
    alpha = 0.01
    weights = np.ones((n,1))
    print 'gradAscent begin'
    numiter=1
    for j in range(numiter):
        print "numiter:", j
        dataIndex=range(m)
        for i in range(m):
            alpha = 4.0/(1.0+j+i)+0.01
            randIndex = int(np.random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(data[randIndex]*weights))
            err = float(labels[randIndex]==label) - h
            weights = weights + alpha * (data[randIndex].T) * err
            del(dataIndex[randIndex])
    return weights

weights=[]
for i in range(10):
    weights.append(gradAscent(dbn1.predict(X),Y,i))

f=open('weights.txt','w')
f.write(str(weights))
f.close()

def test():
    sum1 = [0,0,0,0,0,0,0,0,0,0]
    sum2 = [0,0,0,0,0,0,0,0,0,0]
    index = 0
    for i in dbn1.predict(X_test):
        temppre = -1000
        tempY = -1000
        for j in range(10):
            pre = i*weights[j]
            if pre > temppre:
                temppre = pre
                tempY = j
        if tempY==Y_test[index]:
            sum1[Y_test[index]]=sum1[Y_test[index]]+1
        sum2[Y_test[index]]=sum2[Y_test[index]]+1
        index = index + 1
    for i in range(10):
        print float(sum1[i])/float(sum2[i]),' , ',sum1[i],'/',sum2[i]

test()

im=Image.open('2.bmp')
t=np.array(im)
tt = []
for i in t:
    for j in i:
        tt.append(j)
print 'tt', tt
target = []
target.append(tt)
target = np.asarray(target,'float32')
target = target/255.0  # 0-1 scaling

temppre = -100
tempY = -100
result = []
for j in range(10):
    pre = dbn1.predict(target)*weights[j]
    result.append(pre)
    if pre > temppre:
        temppre = pre
        tempY = j
        print tempY
print result