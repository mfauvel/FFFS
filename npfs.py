# -*- coding: utf-8 -*-
'''
Created on 12 déc. 2014

@author: mfauvel
'''
import scipy as sp
from scipy import linalg
import multiprocessing as mp


## Utilitary functions
def mylstsq(a, b, rcond):
    '''
    Compute a safe/fast least square fitting. 
    For that particular case, the number of unknown parameters is equal to the number of equations. 
    However, a is a covariance matrix that might estimated with a number of samples less than the number of variables, leading to a badly conditionned covariance matrix.
    So, the rcond number is check: if it is ok, we use the fast linalg.solve function; otherwise, we use the slow, but safe, linalg.lstsq function.
    Inputs:
    a: a symmetric definite positive matrix d times d
    b: a d times n matrix
    Outputs:
    x: a d times n matrix
    '''
    eps = sp.finfo(sp.float64).eps
    if rcond>eps: # If the condition number is not too bad try linear system
        try:
            x = linalg.solve(a,b)
        except linalg.LinAlgError: # If error, use least square estimations
            x = linalg.lstsq(a,b)[0]
    else:# Use least square estimate
        x = linalg.lstsq(a,b)[0]
    return x

def safe_logdet(cov):
    '''
    The function computes a secure version of the logdet of a covariance matrix and it returns the rcondition number of the matrix.
    Inputs:
        cov
    Outputs:
        the logdet
    '''
    eps = sp.finfo(sp.float64).eps
    e = linalg.eigvalsh(cov)
    if e.max()<eps:
        rcond = 0
    else:
        rcond = e.min()/e.max()    
    e = sp.where(e<eps,eps,e)    
    return sp.sum(sp.log(e)),rcond

def compute_JFD(v,model,x,y,ids,ind):
    '''
    '''
    # Get parameters
    C=int(y.max())
    ids.append(v)
    JM = 0.0
    for i in range(C):
        for j in range(i+1,C):
            md = (model.mean[i,ids]-model.mean[j,ids])
            cs = (model.cov[i,ids,:][:,ids]+model.cov[j,ids,:][:,ids])/2
            di = linalg.det(model.cov[i,ids,:][:,ids])
            dj = linalg.det(model.cov[j,ids,:][:,ids])
            dij = linalg.det(cs)
            bij = sp.dot(md,linalg.solve(cs,md))/8 + 0.5*sp.log(dij/sp.sqrt(di*dj))
            JM += sp.sqrt(2*(1-sp.exp(-bij)))*model.prop[i]*model.prop[j]
    JM*=2
    ids.pop()
    return ind,JM

def regularization_predict(tau,model,xT,yT):
    """ Function that computes the prediction of the GMM for the cross validation. 
        It will be used together with pool.apply to be applied in parallel mode
        Input:
            model : the GMM mode learns without regularization
            xT : the samples to be classified
            yT : the label
            tau : the regularization parameter
            j : the value 
        Output:
            err : the classification error
            
        Used in GMM.cross_validation()
    """
    d=xT.shape[1]
    err = sp.zeros(tau.size)
    for j,t in enumerate(tau):
        model.tau=t*sp.eye(d)
        yp = model.predict_gmm(xT)[0]  # Predict for the ith fold
        eq = sp.where(yp.ravel()==yT.ravel())[0]
        err[j] = eq.size*100.0/yT.size
        model.tau = None
    return err

def compute_v_cv_gmm(variable,model_cv,xt,yt,ids):
    """ Function that computes the accuracy of the model_cv using the variable : variable + ids
        Inputs:
            variable: the variable to add to ids
            model_cv: the model build with all the variables
            xt,yt: the samples/label for testing
            ids: the pool of retained variables
        Output:
            err: the estimated error
    
        Used in GMM.forward_selection()
    """
    err = sp.zeros(variable.size)
    
    for j,var in enumerate(variable):
        id_t = list(ids)
        id_t.append(var)
        id_t.sort()      
        yp = model_cv.predict_gmm(xt,ids=id_t)[0] # Use the marginalization properties to update the model for each tuple of variables
        eq = sp.where(yp.ravel()==yt.ravel())[0]
        err[j] = float(eq.size)/yp.size
        del id_t

    return err

def compute_loocv_gmm(variable,model,x,y,ids,K_u,alpha,beta,log_prop_u):
    """ Function that computes the estimation of the loocv for the GMM model with variables ids + variable(i)
        Inputs:
            model : the GMM model
            x,y : the training samples and the corresponding label
            ids : the pool of selected variables
            variable   : the variable to be tested from the set of available variable
            K_u    : the initial prediction values computed with all the samples
            alpha, beta and log_prop_u : constant that are computed outside of the loop to increased speed
        Outputs:
            loocv_temp : the loocv
            
        Used in GMM.forward_selection()
    """
    n = x.shape[0]
    ids.append(variable)      # Iteratively add one of the remaining variables
    Kp = model.predict_gmm(x,ids=ids)[1]# Predict with all the samples with ids
    loocv_temp=0.0;                     # Initialization of the temporary loocv
    for j in range(n):                  # Predict the class with the model ids_t
        Kloo = Kp[j,:] + K_u  # Initialization of the decision rule for sample "j" #--- Change for only not C---#
                   
        c = int(y[j]-1)        # Update of parameter of class c
        m  = (model.ni[c]*model.mean[c,ids] -x[j,ids])*alpha[c]    # Update the mean value
        xb = x[j,ids] - m                                     # x centered
        cov_u =  (model.cov[c,ids,:][:,ids] - sp.outer(xb,xb)*alpha[c])*beta    # Update the covariance matrix 
        logdet,rcond = safe_logdet(cov_u)
        Kloo[c] =   logdet - 2*log_prop_u[c] + sp.vdot(xb,mylstsq(cov_u,xb.T,rcond))    # Compute the new decision rule
        del cov_u,xb,m,c                   
                    
        yloo = sp.argmin(Kloo)+1
        loocv_temp += float(yloo==y[j])                   # Check the correct/incorrect classification rule
    ids.pop()                                                         # Remove the current variable 
    return loocv_temp/n                                           # Compute loocv for variable 

class CV:
    '''
    This class implements the generation of several folds to be used in the cross validation
    '''
    def __init__(self):
        self.it=[]
        self.iT=[]

    def split_data(self,n,v=5):
        ''' The function split the data into v folds. Whatever the number of sample per class
        Input:
            n : the number of samples
            v : the number of folds
        Output: None        
        '''
        step = n //v  # Compute the number of samples in each fold
        sp.random.seed(1)   # Set the random generator to the same initial state
        t = sp.random.permutation(n)    # Generate random sampling of the indices
        
        indices=[]
        for i in range(v-1):            # group in v fold
            indices.append(t[i*step:(i+1)*step])
        indices.append(t[(v-1)*step:n])
                
        for i in range(v):
            self.iT.append(sp.asarray(indices[i]))
            l = range(v)
            l.remove(i)
            temp = sp.empty(0,dtype=sp.int64)
            for j in l:            
                temp = sp.concatenate((temp,sp.asarray(indices[j])))
            self.it.append(temp)

    def split_data_class(self,y,v=5):
        ''' The function split the data into v folds. The samples of each class are split approximatly in v folds
        Input:
            n : the number of samples
            v : the number of folds
        Output: None
        '''
        # Get parameters
        n = y.size
        C = y.max().astype('int')
       
        # Get the step for each class
        tc = []
        for j in range(v):
            tempit = []
            tempiT = []
            for i in range(C):
                # Get all samples for each class
                t  = sp.where(y==(i+1))[0]
                nc = t.size
                stepc = nc // v # Step size for each class
                if stepc == 0:
                    print "Not enough sample to build "+ str(v) +" folds in class " + str(i)                                    
                sp.random.seed(i)   # Set the random generator to the same initial state
                tc = t[sp.random.permutation(nc)] # Random sampling of indices of samples for class i
                        
                # Set testing and training samples
                if j < (v-1):
                    start,end = j*stepc,(j+1)*stepc
                else:
                    start,end = j*stepc,nc
                tempiT.extend(sp.asarray(tc[start:end])) #Testing
                k = range(v)
                k.remove(j)
                for l in k:
                    if l < (v-1):
                        start,end = l*stepc,(l+1)*stepc
                    else:
                        start,end = l*stepc,nc
                    tempit.extend(sp.asarray(tc[start:end])) #Training

            self.it.append(tempit)
            self.iT.append(tempiT)
            
## Gaussian Mixture Model (GMM) class
class GMM:

    def __init__(self,size=None,d=None):
        if size is None:
            self.ni = []
            self.prop = []
            self.mean = []
            self.cov =[]
            self.tau = None
            self.ids = None
        else:
            self.ni = sp.empty((size,1))    # Vector of number of samples for each class
            self.prop = sp.empty((size,1))  # Vector of proportion
            self.mean = sp.empty((size,d))  # Vector of means
            self.cov = sp.empty((size,d,d)) # Matrix of covariance
            
    def learn_gmm(self,x,y,tau=None):
        '''
        Function that learns the GMM from training samples
            It is possible to add a regularizer term Sigma = Sigma + tau*I 
        Input:
            x : the training samples
            y :  the labels
            tau : the value of the regularizer, if tau = None (default) no regularization
        Output:
            the mean, covariance and proportion of each class
        '''
        ## Get information from the data
        C = int(y.max(0))   # Number of classes
        n = x.shape[0]  # Number of samples
        d = x.shape[1]  # Number of variables
        
        ## Initialization
        self.ni = sp.empty((C,1))    # Vector of number of samples for each class
        self.prop = sp.empty((C,1))  # Vector of proportion
        self.mean = sp.empty((C,d))  # Vector of means
        self.cov = sp.empty((C,d,d)) # Matrix of covariance
        
        ## Learn the parameter of the model for each class
        for i in range(C):
            j = sp.where(y==(i+1))[0]
            self.ni[i] = float(j.size)    
            self.prop[i] = self.ni[i]/n
            self.mean[i,:] = sp.mean(x[j,:],axis=0)
            self.cov[i,:,:] = sp.cov(x[j,:],bias=1,rowvar=0)  # Normalize by ni to be consistent with the update formulae
        if tau is not None:
            self.tau = tau*sp.eye(d)
            
    def predict_gmm(self,xt,ids=None):
        '''
        Function that predict the label for sample xt using the learned model
        Inputs:
            xt: the samples to be classified
        Outputs:
            y: the class
            K: the decision value for each class      

        '''
        ## Get information from the data
        nt = xt.shape[0]        # Number of testing samples
        C = self.ni.shape[0]    # Number of classes
        
        ## Initialization
        K = sp.empty((nt,C))
        
        ## Start the prediction for each class
        for c in range(C):
            if ids is None: # Predict with all the features
                xtc = xt-self.mean[c,:]
                if self.tau is None: # Nothing to do 
                    logdet,rcond = safe_logdet(self.cov[c,:,:]) 
                    cst = logdet -2*sp.log(self.prop[c]) # Pre compute the constant term
                    temp = mylstsq(self.cov[c,:,:],xtc.T,rcond).T
                else: # We need to add the regularization
                    logdet,rcond = safe_logdet(self.cov[c,:,:]+self.tau) 
                    cst = logdet -2*sp.log(self.prop[c]) # Pre compute the constant term
                    temp = mylstsq(self.cov[c,:,:]+self.tau,xtc.T,rcond).T                            
                K[:,c] = sp.sum(xtc*temp,axis=1)+cst
            else:
                logdet,rcond = safe_logdet(self.cov[c,ids,:][:,ids])
                cst =  logdet - 2*sp.log(self.prop[c]) # Pre compute the constant term
                xtc = xt[:,ids] - self.mean[c,ids]
                temp = mylstsq(self.cov[c,ids,:][:,ids],xtc.T,rcond).T
                K[:,c] = sp.sum(xtc*temp,axis=1)+cst
            del temp,xtc
            
        ## Assign the label to the minimum value of K 
        yp = sp.argmin(K,1)+1
        return yp,K
    
    def cross_validation(self,x,y,tau,v=5,ncpus=None):
        ''' 
        Function that computes the cross validation accuracy for the value tau of the regularization
        Input:
            x : the training samples
            y : the labels
            tau : a range of values to be tested
            v : the number of fold
        Output:
            err : the estimated error with cross validation for all tau's value
        '''
        ns = x.shape[0]     # Number of samples
        np = tau.size       # Number of parameters to test
        cv = CV()           # Initialization of the indices for the cross validation
        cv.split_data_class(y)
        err = sp.zeros(np)  # Initialization of the error
        if ncpus is None:        
            ncpus=mp.cpu_count()    # Get the number of core
        
        ## Create GMM model for each fold
        model_cv = []
        for i in range(v):
            model_cv.append(GMM())
            model_cv[i].learn_gmm(x[cv.it[i],:], y[cv.it[i]])
        
        pool = mp.Pool(processes=ncpus)
        processes =  [pool.apply(regularization_predict, args=(tau,model_cv[i],x[cv.iT[i],:],y[cv.iT[i]])) for i in range(v)]
        pool.close()
        pool.join()
        
        for p in processes:
            err += p
        
        ## Free memory        
        for model in model_cv:
            del model
        
        del processes,pool,model_cv

        return err/v
    
    def forward_selection(self,x,y,delta=0.1,maxvar=None,v=5,ncpus=None):
        """ Function that selects the most discriminative variables according to a forward search
            Inputs:
                x,y :  the training samples and their labels
                delta :  the minimal improvement in percentage when a variable is added to the pool, the algorithm stops if the improvement is lower than delta. Default value 0.1%
                maxvar: maximum number of extracted variables. Default valule: 20% of the origianl number
                v: number of folds for the cross-validation. Default value: None -> do loocv and use fast estimation of the updated model. Otherwise, do fold-fold cross-validation with conventionnal learning of the model
                ncpus=
                               
            Outputs:
                ids: the selected variable
                OA: the accuracy estimated for each ids by loocv or v-fold cv
        """
        ## Get some information from the variable
        C = int(y.max(0));  # Number of classes
        n = x.shape[0]      # Number of samples
        d = x.shape[1]      # Number of variables
        if ncpus is None:
            ncpus=mp.cpu_count()# Get the number of core
        
        ## Initialization
        r=0                 # Initialization of the counter
        variable = sp.arange(d) # At step zero: d variables available
        ids=[]              # and no selected variable
        OA=[]              # list of the evolution the OA estimation                
        if maxvar is None:
            maxvar = sp.floor(d/5)  # Select at max 20 % of the original number of variables
            
        if v is None: # LOOCV estimation of the error
            ## Precompute the proportion and some others constants
            log_prop_u = [sp.log((n*self.prop[c]-1.0)/(n-1.0)) for c in range(C)]
            K_u = 2*sp.log((n-1.0)/n)
            beta = self.ni[c]/(self.ni[c]-1.0)              # Constant for the rank one downdate
            alpha = [1/(self.ni[c]-1) for c in range(C)] 
        
            ## Start the forward search
            while(r<maxvar):            
                loocv = sp.zeros(variable.size)
                pool = mp.Pool(processes=ncpus)
                processes = [pool.apply(compute_loocv_gmm,args=(v,self,x,y,ids,K_u,alpha,beta,log_prop_u)) for v in variable]
                pool.close()
                pool.join()
                for i,p in enumerate(processes):
                    loocv[i] = p                
                        
                ## Select the variable that provides the highest loocv
                t = sp.argmax(loocv)                # get the indice of the maximum of loocv
                OA.append(loocv[t])                # add the value to loo
                if r==0:
                    ids.append(variable[t])         # add the selected variable to the pool
                    variable=sp.delete(variable,t)  # remove the selected variable from the initial set
                elif (variable.size == 0) or (((OA[r]-OA[r-1])/OA[r-1]*100) < delta):
                    OA.pop()
                    break
                else:
                    ids.append(variable[t])
                    variable=sp.delete(variable,t)
                r =r+1
        
        else:
            cv=CV()                                 # Initialize the CV sets
            cv.split_data_class(y,v=v)              # Generate split indices for the data
            
            ## Pre-update the models
            model_pre_cv = []
            for i in range(v):
                model_pre_cv.append(GMM(size=C,d=d))# List of updated GMM models
                X,Y=x[cv.iT[i],:], y[cv.iT[i]]
                nu = float(Y.size)
                for j in range(C):                  #Update the model for each class
                    k = sp.where(Y==(j+1))[0]
                    nu_c = float(k.size)
                    mean_t = sp.mean(X[k,:],axis=0)
                    cov_t = sp.cov(X[k,:],bias=1,rowvar=0)
                        
                    model_pre_cv[i].ni[j] = self.ni[j]-nu_c
                    model_pre_cv[i].prop[j]= model_pre_cv[i].ni[j]/(n-nu)
                    model_pre_cv[i].mean[j,:] = (self.ni[j]*self.mean[j,:]-nu_c*mean_t)/(self.ni[j]-nu_c) 
                    model_pre_cv[i].cov[j,:] = (self.ni[j]*self.cov[j,:,:] - nu_c*cov_t - nu_c*self.ni[j]/model_pre_cv[i].ni[j]*sp.outer(self.mean[j,:]-mean_t,self.mean[j,:]-mean_t))/model_pre_cv[i].ni[j]
                    del k,nu_c,mean_t,cov_t
                del X,Y,nu
                
            ## Start the forward search
            while(r<maxvar):
                err=sp.zeros(variable.size)
                pool = mp.Pool(processes=ncpus)     
                processes =  [pool.apply_async(compute_v_cv_gmm, args=(variable,model_pre_cv[i],x[cv.iT[i],:],y[cv.iT[i]],ids)) for i in xrange(v)]
                pool.close()
                pool.join()
                for p in processes:
                    err += p.get()
                err /= v
                del processes,pool                
                ## Select the variable that provides the highest loocv
                t = sp.argmax(err)                # get the indice of the maximum of loocv
                OA.append(err[t])                # add the value to loo
                if r==0:
                    ids.append(variable[t])         # add the selected variable to the pool
                    variable=sp.delete(variable,t)  # remove the selected variable from the initial set
                elif (variable.size == 0) or (((OA[r]-OA[r-1])/OA[r-1]*100) < delta):
                    OA.pop()
                    break
                else:
                    ids.append(variable[t])
                    variable=sp.delete(variable,t)
                r += 1
                   
        ## Return the final value
        return ids,OA

    def forward_selection_JM(self,x,y,delta=0.1,maxvar=None,ncpus=None):
        '''
        '''
        ## Get some information from the variable
        C = int(y.max(0));  # Number of classes
        n = x.shape[0]      # Number of samples
        d = x.shape[1]      # Number of variables
        if ncpus is None:
            ncpus=mp.cpu_count()# Get the number of core
        
        ## Initialization
        r=0                 # Initialization of the counter
        variable = sp.arange(d) # At step zero: d variables available
        ids=[]              # and no selected variable
        JMD=[]              # list of the evolution the OA estimation                
        if maxvar is None:
            maxvar = sp.floor(d/5)  # Select at max 20 % of the original number of variables

        while(r<maxvar):
            JMd = sp.zeros(variable.size)
            pool = mp.Pool(processes=ncpus)
            processes = [pool.apply_async(compute_JFD,args=(v,self,x,y,ids,ind)) for ind,v in enumerate(variable)]
            pool.close()
            pool.join()
            for p in processes:
                ind,jm = p.get()
                JMd[ind] = jm
            
            ## Select the variable that provides the highest loocv
            t = sp.argmax(JMd)                # get the indice of the maximum of loocv
            JMD.append(JMd[t])                  # add the value to loo
            if r==0:
                    ids.append(variable[t])         # add the selected variable to the pool
                    variable=sp.delete(variable,t)  # remove the selected variable from the initial set
            elif (variable.size == 0) or (((JMD[r]-JMD[r-1])/JMD[r-1]*100) < delta):
                JMD.pop()
                break
            else:
                ids.append(variable[t])
                variable=sp.delete(variable,t)
            r =r+1
        
        ## Return the final value
        return ids,JMD
