
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import cvxopt
from matplotlib import cm
#import xlrd
#uncooment for processing data from excel
#wb=xlrd.open_workbook("collegeappsdat.xlsx")
#sh = wb.sheet_by_index(0)
#check = sh.row_values(1)



#data set to check SVM with
data = [[3.54, 2130.0, 33.0, 'Accepted']
,
[3.97, 2230.0, '-', 'Accepted']
,
[3.52, 2035.0, 35.0, 'Accepted']
,
[3.65, 2250.0, '-', 'Accepted']
,
[3.95, 2050.0, '-', 'Accepted']

]


#format data for Support Vector Machine
#support vector machine only uses -1 and 1 
#not accpeted is -1 and accepted is 1
xandy  = []
acceptance = []
for i in range(0,len(data)):
    if data[i][1] != "-":
        xandy.append(data[i][0:2])
        if data[i][3] == "Accepted":
            acceptance.append(1.0)
        else:
            acceptance.append(-1.0)
#print(xandy)
#print(acceptance)

#http://www.tristanfletcher.co.uk/SVM%20Explained.pdf

def polynomialK(u,v,b):
    return (np.dot(u,v)+b)**2    
    

def gaussianK(v1, v2, sigma):
    return np.exp(-np.linalg.norm(v1-v2, 2)**2/(2*sigma**2))

class SVM:
    def __init__(self,X,y):
        self.X = X
        self.y = y
        
    def findParameters(self,X,y):
        # min 1/2 x^T P x + q^T x
        #Ax = b
        #y's are if the student is accepted -1 for being not accpeted 1 for being accpeted 
        #put in cvxopt 
        #print(y)
        n_samples, n_features = X.shape
        K = self.gramMatrix(X)
        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(-1 * np.ones(n_samples))
        #Gtry = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
        #print(Gtry)
        #htry = cvxopt.matrix(np.zeros(n_samples))
        
        G_std = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
        h_std = cvxopt.matrix(np.zeros(n_samples))

        G_slack = cvxopt.matrix(np.diag(np.ones(n_samples)))
        #I'm replacing self.c with a constant 
        h_slack = cvxopt.matrix(np.ones(n_samples) * 20)

        G = cvxopt.matrix(np.vstack((G_std, G_slack)))
        h = cvxopt.matrix(np.vstack((h_std, h_slack)))

        
        A = cvxopt.matrix(y, (1, n_samples))
        
        b = cvxopt.matrix(0.0)

        param = cvxopt.solvers.qp(P, q, G, h, A, b)
        array = param['x']
        return array
    
    
    def WB_calculator(self,X,y):
        #calculates w vector
        yi = self.y
        X = np.asarray(X)
        y = np.asarray(y)
        important = self.findParameters(X,y)
        #print("these are parameters")
        #print(important)
        firstsum = [0 for x in range(0,len(y))]
        for point in range(0,len(important)):
            liste = X[point]*important[point]*yi[point]
            firstsum = [x + y for x, y in zip(firstsum,liste)]
            

            
        #this part calculates bias
            #this is a very naive implementation of bias
            #xstuff is the x_coordinate vector we find this by transpose
            #bunsen is the sum of bias
            bunsen = 0
        for i in range(0,len(important)):
            bunsen = bunsen+ (yi[i]- np.dot(firstsum,X[i]))
            
        avgB = bunsen/len(important)
        answer = (firstsum , avgB)
        #print("w vector")
        
        #print(firstsum)
        return answer  
    
    
#computes the gramMatrix given a set of all points included in the data
#this is basicly a matrix of dot products 
    def gramMatrix(self,X): 
        gramMatrix = []
        data = np.asarray(self.X)
        dataTran = data
        for x in dataTran:
            row = []
            for y in dataTran:
               
                row.append(np.dot(x,y))
                
            gramMatrix.append(row)
          
        return gramMatrix
    
    
    
    #determine accepted or not base on array point and
    #input and output matrix
    def detAcceptance(self,point,X,y):
        #calculates weights that will be used
        cutoff = self.WB_calculator(X,y)
        if(np.dot(cutoff[0],point)+cutoff[1] >0):
            print("You got in")
        elif(np.dot(cutoff[0],point)+cutoff[1]<0):
            print("Study")
                
    # plots plane and points in 3D
     # plots ACT ,SAT ,and GPA           
    def DGraph(self,X,y):
        important_stuff = self.WB_calculator(X,y)
        weights = important_stuff[0] 
        c = important_stuff[1]
        #here we actaually graph the functionb 
        graphable = X.T
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        xs = graphable[0]
        ys = graphable[1]
        zs = graphable[2]
        
        colors = self.y
        ax.scatter(xs,ys,zs,c=colors)
        ax.set_xlabel("A")
        ax.set_ylabel("B")
        ax.set_zlabel("C")
        #this changes orientation and look of surface
        ax.view_init(azim = 180+160,elev = 0)
        X = np.arange(-1.2, 1.2, 0.25)
        Y = np.arange(-1.2, 1.2, 0.25)
        X, Y = np.meshgrid(X, Y)
        
        Z = ((-weights[0]*X + -weights[1]*Y - c)/(weights[2]))
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=True)
        plt.show()
        
        
        #2D graph of SAT and GPA data
    def TWDGraph(self,X,y):
        important_stuff = self.WB_calculator(X,y)
        weights = important_stuff[0]
        w1 = weights[0]
        w2 = weights[1]
        c = important_stuff[1]
        print(c)
        print(w2)
        figure2 = plt.figure();
        #setting up coordinates
        graphable = X.T
        xaxis = [x for x in graphable[0]]
        yaxis = [x for x in graphable[1]]
        print("done")
        #graph dividing line
        fig, ax = plt.subplots()
        ax.scatter(xaxis,yaxis)
        #X = np.arange(-2, 2,.1)
        x_min, x_max = [0,1]
        y_min, y_max = [(-c/w2) , ((-c/w2)+ (-w1/w2)*(x_max-x_min))]
        ax.plot([x_min, x_max], [y_min, y_max])
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        plt.show()
        




    
#testing section uncomment to test 
              
        
#Readydat = np.asarray(xandy)
#Readydat = Readydat/np.amax(Readydat,axis=0)
#Readyacc = np.asarray(acceptance)
#d = SVM(Readydat,Readyacc)
#d.TWDGraph(Readydat, Readyacc)
    
#print(d.detAcceptance([2400,2],Readydat, Readyacc))
    
#second testing 
    
    #tryit = SVM([1,2],[3,1])
    #d.TWDGraph(Readydat, Readyacc)
    
    #d.DGraph(Readydat, Readyacc)
    #print(len(Readydat))
    #print(len(Readyac))        
#a = [[.1,.1,.1],[.2,.2,.2],[.15,.15,.15],[.9,.9,.9],[.95,.95,.95]]
#a2 = [[.1,.1],[.2,.2],[.9,.9],[20,20],[30,30]]
#scaling array
#Hans = np.asarray(a2)
#Hans = Hans/np.amax(Hans,axis=0)
#acceptance list
#b = [-1.0,-1,-1,1,1]
#bigger =np.asarray(b)
#d = SVM(Hans,bigger)
#d.TWDGraph(Hans,bigger)
#d.TWDGraph(xandy, acceptance)
#print(d.gramMatrix(check)[0])
#print("parameters ya")
#print(d.findParameters(check,bigger))
#print(d.WB_calculator(Hans,bigger))
#d.Graph(Hans,bigger)
#determines acceptance for prospective applicant
#print(d.detAcceptance([.01,.01],Hans,bigger))
     
        
        
        
        
        
        
        
        
        
        
        
        

        
