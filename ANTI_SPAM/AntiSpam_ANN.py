
import numpy as np 
from scipy import optimize
class Anti_SPAM_ANN_Approach(object):
    def __init__(self, Lambda=0):        
        #Define Hyperparameters
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3
        
        #Weights (parameters)
        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)
        
        #Regularization Parameter:
        self.Lambda = Lambda
        
    def forward(self, X):
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3) 
        return yHat
        
    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))
    
    def sigmoidPrime(self,z):
        #Gradient of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)
    
    def costFunction(self, X, y):
        #Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2)/X.shape[0] + (self.Lambda/2)*(np.sum(self.W1**2)+np.sum(self.W2**2))
        return J
        
    def costFunctionPrime(self, X, y):
        #Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)
        
        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        #Add gradient of regularization term:
        dJdW2 = np.dot(self.a2.T, delta3)/X.shape[0] + self.Lambda*self.W2
        
        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        #Add gradient of regularization term:
        dJdW1 = np.dot(X.T, delta2)/X.shape[0] + self.Lambda*self.W1
        
        return dJdW1, dJdW2
    
    #Helper functions for interacting with other methods/classes
    def getParams(self):
        #Get W1 and W2 Rolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params
    
    def setParams(self, params):
        #Set W1 and W2 using single parameter vector:
        W1_start = 0
        W1_end = self.hiddenLayerSize*self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], \
                             (self.inputLayerSize, self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], \
                             (self.hiddenLayerSize, self.outputLayerSize))
        
    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))



class trainer(object):
    def __init__(self, N):
        #Make Local reference to network:
        self.N = N
        
    def callbackF(self, params):
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.X, self.y))   
        
    def costFunctionWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.costFunction(X, y)
        grad = self.N.computeGradients(X,y)
        
        return cost, grad
        
    def train(self, X, y):
        #Make an internal variable for the callback function:
        self.X = X
        self.y = y

        #Make empty list to store costs:
        self.J = []
        
        params0 = self.N.getParams()

        options = {'maxiter': 200, 'disp' : True}
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS', \
                                 args=(X, y), options=options, callback=self.callbackF)

        self.N.setParams(_res.x)
        self.optimizationResults = _res
#Train_DATA 
X=np.array(([0.16129032258064513, 0.31875], [0.19354838709677424, 0.3322222222222222], [0.12903225806451613, 0.4130081300813008], [0.16129032258064513, 0.31749999999999995], [0.19354838709677424, 0.17843137254901958], [0.19354838709677424, 0.15069444444444444], [0.16129032258064513, 0.28385826771653544], [0.12903225806451613, 0.175], [0.12903225806451613, 0.19209302325581395], [0.12903225806451613, 0.12727272727272726]),dtype=float)
Y=np.array(([0],[0],[0],[0],[1],[1],[1],[1],[1],[1]),dtype=float)

#Test_DATA 
XT=np.array(([0.16129032258064513, 0.175],
 [0.12903225806451613, 0.182],
 [0.12903225806451613, 0.23890243902439023]),dtype=float)
YT=np.array(([1],[1],[1]),dtype=float)

#Scaling of Data 
X=X/np.amax(X,axis=0)
Y=Y/1

NN=Anti_SPAM_ANN_Approach()
NN.forward(X)

NN.forward(X)

T=trainer(NN)

T.train(X,Y)


Y


NN.forward(X)

T.train(X,Y)

NN.forward(X)

Y

NN.forward(XT)

import os
import dns.resolver
import sys
import socket
list_train=os.listdir(r'C:\Users\asus\Desktop\Etudes\S4\ProjetAI\train')
list_test=os.listdir(r'C:\Users\asus\Desktop\Etudes\S4\ProjetAI\test')
list_train
list_test
file=r'C:\Users\asus\Desktop\Etudes\S4\ProjetAI\jargon.txt'

def tokenize_text(list_train):
	dir=r'C:\Users\asus\Desktop\Etudes\S4\ProjetAI\train'
	Gtoken=[]
	for file in list_train:
		token=[]  
		chaine="  "
		f = open(dir+'\\'+file, 'r')
		token.append(file)
		for line in f.readlines():
			if len(token)<=4: 
				token.append(line) 
			else:
				chaine=chaine+line 
		token.append(chaine)                 
		f.close()
		Gtoken.append(token)
	return Gtoken
p=tokenize_text(list_train)
def tokenize_text(list_test):
	dir=r'C:\Users\asus\Desktop\Etudes\S4\ProjetAI\test'
	Gtoken=[]
	for file in list_test:
		token=[]  
		chaine="  "
		f = open(dir+'\\'+file, 'r')
		token.append(file)
		for line in f.readlines():
			if len(token)<=4: 
				token.append(line) 
			else:
				chaine=chaine+line 
		token.append(chaine)                 
		f.close()
		Gtoken.append(token)
	return Gtoken
p=tokenize_text(list_test)
 
def Listing_RBL(x): 
	open("data.txt", "w").close()    
	bls = ["zen.spamhaus.org", "spam.abuse.ch", "cbl.abuseat.org", "virbl.dnsbl.bit.nl", "dnsbl.inps.de", 
	"ix.dnsbl.manitu.net", "dnsbl.sorbs.net", "bl.spamcannibal.org", "bl.spamcop.net", 
	"xbl.spamhaus.org", "pbl.spamhaus.org", "dnsbl-1.uceprotect.net", "dnsbl-2.uceprotect.net", 
	"dnsbl-3.uceprotect.net", "db.wpbl.info","all.s5h.net","b.barracudacentral.org","bl.emailbasura.org",
"bl.spamcannibal.org","bl.spamcop.net","blacklist.woody.ch",
"bogons.cymru.com","cbl.abuseat.org","cdl.anti-spam.org.cn",
"combined.abuse.ch","db.wpbl.info","dnsbl-1.uceprotect.net","wormrbl.imp.ch","xbl.spamhaus.org","z.mailspike.net",
"zen.spamhaus.org"]
	data = socket.gethostbyname(x)
	myIP =data
	length=len(bls)
	print (myIP)
	for bl in bls:
		try:
			my_resolver = dns.resolver.Resolver()
			query = '.'.join(reversed(str(myIP).split("."))) + "." + bl
			answers = my_resolver.query(query, "A")
			answer_txt = my_resolver.query(query, "TXT" )
			print ('IP: %s IS listed in %s (%s: %s)' %(myIP, bl, answers[0], answer_txt[0]))
		except dns.resolver.NXDOMAIN:
			fichier = open("data.txt", "a+")
			fichier.write('NOT_LISTED \n')
		except socket.gaierror: 
			print ("")
		except dns.resolver.NoNameservers:
			print ("")
		except dns.resolver.Timeout: 
			print ("")
def count_file_line():
    f = open("data.txt","r")
    l=[]
    for line in f.readlines():
        l.append(line)
    f.close()
    return len(l)

def metric_FROM(): 
    #la creation du fichier data.txt est obligatoire à ce niveau avant de commencer 
    j=count_file_line()# on retourne de degree d'inquiétude 1-degre de safety (NON LISTER ) 
    return 1-j/31

def extract_domain(chaine): 
    pos=chaine.find('@')
    pos2=chaine.find('>')
    domaine=chaine[pos+1:pos2]
    return domaine 

def metric_SUBJECT_MESSAGE(subject,message,file):
    #traitement subject evalué à 30% 
    f=open(file,"r")
    text=f.read() 
    #text=text.split(',')
    sub=0
    mes=0
    subject=subject.split(' ')
    message=message.split(' ')
    
    for x in subject: 
        if x in text : 
            sub=sub+1
    for x in message: 
        if x in text : 
            mes=mes+1
    sub=sub/len(subject) 
    mes=mes/len(message) 
    res=sub*0.300 + mes*0.700 
    #traitement subject evalué à 70% 
   
    f.close()
    return res

file=r'C:\Users\asus\Desktop\Etudes\S4\ProjetAI\jargon.txt'
def Quantify(list_train): 
    p=tokenize_text(list_train)
    list=[]
    for x in p : 
        item=[]
        domaine=extract_domain(x[2])
        Listing_RBL(domaine)
        item.append(x[0])
        item.append(metric_FROM())
        item.append(metric_SUBJECT_MESSAGE(x[4],x[5],file)) 
        list.append(item)
    return list 




p=Quantify(list_test)


def np_array_format(p):
    list=[]
    for x in p: 
        list.append([x[1],x[2]])
    return list

      

p
o=np_array_format(p)
o
