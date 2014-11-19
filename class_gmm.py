'''
Created on 25 avr. 2014

@author: mfauvel
'''
import scipy as sp
from scipy import linalg
import multiprocessing as mp
from functools import partial

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
    if rcond>eps:
        x = linalg.solve(a,b)
    else:
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
        rcond = eps
    else:
        rcond = e.min()/e.max()    
    e = sp.where(e<eps,eps,e)    
    return sp.sum(sp.log(e)),rcond

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
    id_t = ids[:]
    id_t.append(variable)
    id_t.sort()
    model_cv_id = model_cv.update(ids=id_t)      # Use the marginalization properties to update the model for each tuple of variables
    yp = model_cv_id.predict_gmm(xt[:,id_t])[0]
    eq = sp.where(yp.ravel()==yt.ravel())[0]
    err = float(eq.size)/yp.size
    del model_cv_id,id_t
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
    Kp = model.predict_gmm_id(x,ids)    # Predict with all the samples with ids
    loocv_temp=0.0;                     # Initialization of the temporary loocv
    for j in range(n):                  # Predict the class with the model ids_t
        Kloo = Kp[j,:] + K_u  # Initialization of the decision rule for sample "j" #--- Change for only not C---#
                   
        c = int(y[j]-1)        # Update of parameter of class c
        m  = (model.ni[c]*model.mean[c,ids] -x[j,ids])*alpha[c]    # Update the mean value
        xb = x[j,ids] - m                                     # x centered
        cov_u =  (model.cov[c,ids,:][:,ids] - sp.outer(xb,xb)*alpha[c])*beta    # Update the covariance matrix 
        logdet,rcond = safe_logdet(cov_u)
        Kloo[c] =   logdet - 2*log_prop_u[c] + sp.vdot(xb,mylstsq(cov_u,xb.T),rcond)    # Compute the new decision rule
        del cov_u,xb,m,c                   
                    
        yloo = sp.argmin(Kloo)+1
        loocv_temp += float(yloo==y[j])                   # Check the correct/incorrect classification rule
    ids.pop()                                                         # Remove the current variable 
    return loocv_temp/n                                           # Compute loocv for variable 

def regularization_predict(tau,model,xT,yT):
    """ Function that computes the prediction of the GMM for the cross validation. 
        It will be used together with "partial" to be applied in parallel mode
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
    model_temp = model.update(tau=tau)
    yp = model_temp.predict_gmm(xT)[0]  # Predict for the ith fold
    eq = sp.where(yp.ravel()==yT.ravel())[0]
    err = eq.size*100/yT.size
    return err  

def worker_predict(xt,model,i,V,Q):
    """ The function computes the decision rules for each class under the Gaussian Mixture Model
        Use two queues V and Q to store the results in multiprocessing mode
        Input:
            xt: the testing samples
            model: the GMM learned
            i: the part of the all testing set that is processed
            Q: queue that stores the results
            V: queue that stores the value of "i"
        
        Used in GMM.predict_gmm()
    """
    nt= xt.shape[0]
    C = model.ni.shape[0]
    K = sp.empty((nt,C))
    
    for c in range(C):
        logdet,rcond = safe_logdet(model.cov[c,:,:])
        cst = logdet -2*sp.log(model.prop[c])
        xtc = xt - model.mean[c,:]
        temp = mylstsq(model.cov[c,:,:],xtc.T,rcond).T
        K[:,c] = sp.sum(xtc*temp,axis=1)+cst
        del temp,xtc
    Q.put(K)
    V.put(i)
    
def compute_mahalanobis_distance(c,prop,mean,cov,xt):
    """ The function computes the decision rules for class c under the Gaussian Mixture Model
        Input:
            prop,ni,mean,cov : are the parameter of the model
            xt : the testing samples
        Output:
            K: the decision function
            
        Used in GMM.predict_gmm()
    """
    logdet,rcond = safe_logdet(cov[c,:,:]) 
    cst = logdet -2*sp.log(prop[c]) # Pre compute the constant term
    xtc = xt-mean[c,:]
    temp = mylstsq(cov[c,:,:],xtc.T,rcond).T
    K = sp.sum(xtc*temp,axis=1)+cst
    del temp, xtc
    return K
         
class CV:#Cross_validation 
    def __init__(self):
        self.it=[]
        self.iT=[]
        
    def split_data(self,n,v=5):
        """ The function split the data into v folds. Whatever the number of sample per class
        Input:
            n : the number of samples
            v : the number of folds
        Output: None        
        """
        sp.random.seed(0)   # Set the random generator to the same initial state
        step = int(sp.ceil(float(n)/v)) # Compute the number of samples in each fold
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
            
    
class GMM:# Gaussian Mixture Model

    def __init__(self,size=None,d=None):
        if size is None:
            self.ni = []
            self.prop = []
            self.mean = []
            self.cov =[]
        else:
            self.ni = sp.empty((size,1))    # Vector of number of samples for each class
            self.prop = sp.empty((size,1))  # Vector of proportion
            self.mean = sp.empty((size,d))  # Vector of means
            self.cov = sp.empty((size,d,d)) # Matrix of covariance 
    
    def learn_gmm(self,x,y,tau=None):
        """ Function that learns the GMM from training samples
            It is possible to add a regularizer term Sigma = Sigma + tau*I 
        Input:
            x : the training samples
            y :  the labels
            tau : the value of the regularizer, if tau = None (default) no regularization
        Output:
            the mean, covariance and proportion of each class
        """
        ## Get some information from the data
        C = int(y.max(0))   # Number of classes
        n = x.shape[0]  # Number of samples
        d = x.shape[1]  # Number of variables
        
        ## Initialization
        self.ni = sp.empty((C,1))    # Vector of number of samples for each class
        self.prop = sp.empty((C,1))  # Vector of proportion
        self.mean = sp.empty((C,d))  # Vector of means
        self.cov = sp.empty((C,d,d)) # Matrix of covariance
        
        ## Learn the parameter of the model
        for i in range(C):
            j = sp.where(y==(i+1))[0]
            self.ni[i] = float(j.size)    # Indices starts at zero in Python !
            self.prop[i] = self.ni[i]/n
            self.mean[i,:] = sp.mean(x[j,:],axis=0)
            self.cov[i,:,:] = sp.cov(x[j,:],bias=1,rowvar=0)  # Normalize by ni to be consistent with the update formulae
            if tau is not None:
                self.cov[i,:,:] += tau*sp.eye(d) 
            
            
    def update(self,ids=None,tau=None):
        """ Function that constructs an updated model by adding the regularization term OR according to the variable ids selected with the forward strategy.
        Input:
            ids: the selected variables
        Output
            update_model: the model updated
        """
        update_model = GMM()
        update_model.prop = sp.copy(self.prop)
        update_model.ni = sp.copy(self.ni)
        if ids is None:
            C = self.ni.shape[0]    # Number of classes
            d = self.mean.shape[1]  # Number of variables
            update_model.mean = sp.copy(self.mean)
            update_model.cov = sp.copy(self.cov)
            for i in range(C):
                update_model.cov[i,:,:] += tau*sp.eye(d)            
        else:
            update_model.mean = sp.copy(self.mean[:,ids])
            update_model.cov = sp.copy(self.cov[:,ids,:][:,:,ids])
        return update_model
    
    
    def predict_gmm(self,xt,do_parallel=None):
        """ Function that predict the label for sample xt using the learned model
            Inputs:
                xt: the samples to be classified
            Outputs:
                y: the class
                K: the decision value for each class             
        """
        ## Get some information from the data
        nt = xt.shape[0]        # Number of testing samples
        C = self.ni.shape[0]    # Number of classes
        
        ## Initialization
        K = sp.empty((nt,C))

        if do_parallel is None:
            ## Start the prediction for each class
            for i in range(C):
                K[:,i] = compute_mahalanobis_distance(i,self.prop,self.mean,self.cov,xt)
        else:
            ## Start the prediction for each class in parallel mode
            ncpus=mp.cpu_count()    # Get the number of core
            Q = mp.Queue()          # Initialization of temporary Queue to store the data
            V = mp.Queue()          # Initialization of temporary Queue to store the data
            processes = []
            step = int(sp.ceil(float(nt)/ncpus))    # Get the step size for the estimation
        
            for i in range(ncpus):                  # Start the processing of the data
                if step*(i+1)<=nt:
                    p = mp.Process(target=worker_predict,args=(xt[step*i:step*(i+1),:],self,i,V,Q))
                else:
                    p = mp.Process(target=worker_predict,args=(xt[step*i:nt,:],self,i,V,Q))
                processes.append(p)
                p.start()
            
            for t in range(ncpus): # Try to do a while loop !
                i = V.get()
                if step*(i+1)<=nt:
                    K[step*i:step*(i+1)]=Q.get()
                else:
                    K[step*i:nt] = Q.get()
        
            for p in processes:
                p.join()           
            
        ## Assign the label to the minimum value of K 
        yp = sp.argmin(K,1)+1
        return yp,K

    def predict_gmm_id(self,xt,ids):
        """ Function that predict the label for sample xt using the learned model with only variable ids
            To be use in the function forward_selection
            Inputs:
                xt: the samples to be classified
                ids: the variables to be used in the model
                get_class: optional argument to get the class for each samples (==1) or just the decision function (==0)
            Outputs:
            y: the class
            K: the decision value for each class             
        """
        ## Get some information from the data
        nt = xt.shape[0]        # Number of testing samples
        C = self.ni.shape[0]    # Number of classes
        
        ## Initialization
        K = sp.empty((nt,C))
        
        ## Start the prediction for each class
        for c in range(C): 
            logdet,rcond = safe_logdet(self.cov[c,ids,:][:,ids])
            cst =  logdet - 2*sp.log(self.prop[c]) # Pre compute the constant term
            xtc = xt[:,ids] - self.mean[c,ids]
            temp = mylstsq(self.cov[c,ids,:][:,ids],xtc.T,rcond).T
            K[:,c] = sp.sum(xtc*temp,axis=1)+cst
            del temp, xtc
        return K

    def cross_validation(self,x,y,tau,v=5):
        """ Function that computes the cross validation accuracy for the value tau of the regularization
        Input:
            x : the training samples
            y : the labels
            tau : a range of values to be tested
            v : the number of fold
        Output:
            err : the estimated error with cross validation for all tau's value
        """
        ns = x.shape[0]     # Number of samples
        np = tau.size       # Number of parameters to test
        cv = CV()           # Initialization of the indices for the cross validation
        cv.split_data(ns)
        err = sp.zeros(np)  # Initialization of the error        
        ncpus=mp.cpu_count()    # Get the number of core
        for i in range(v):      # Start the cross validation
            self.learn_gmm(x[cv.it[i],:], y[cv.it[i]])
            partial_f = partial(regularization_predict,model=self,xT=x[cv.iT[i],:],yT=y[cv.iT[i]])
            p = mp.Pool(processes=ncpus)    # Start ncpus worker process
            err += p.map(partial_f,tau)     # Start the worker
            p.close()                       # No more worker -> do the job
            p.join()                        # Wait until the end
            del partial_f
        err /= v
        
        return err
    
    def forward_selection(self,x,y,delta=0.1,maxvar=None,v=None,ncpus=None):
        """ Function that selects the most discriminative variables according to a forward search
            Inputs:
                x,y :  the training samples and their labels
                delta :  the minimal improvement in percentage when a variable is added to the pool, the algorithm stops if the improvement is lower than delta. Default value 0.1%
                maxvar: maximum number of extracted variables. Default valule: 20% of the origianl number
                v: number of folds for the cross-validation. Default value: None -> do loocv and use fast estimation of the updated model. Otherwise, do fold-fold cross-validation with conventionnal learning of the model
                   
                               
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
                partial_f=partial(compute_loocv_gmm,model=self,x=x,y=y,ids=ids,K_u=K_u,alpha=alpha,beta=beta,log_prop_u=log_prop_u) #Compute the loocv estimate for the variable [ids,variable[i]]
                p = mp.Pool(processes=ncpus)        # Start ncpus worker process            
                loocv = p.map(partial_f,variable)   # Start the worker
                p.close()                       # No more worker -> do the job
                p.join()                        # Wait until the end
                del partial_f
        
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
            cv.split_data(n,v=v)                    # Generate split indices for the data
            
            # Pre-update the models
            model_pre_cv = []
            for i in range(v):
                model_pre_cv.append(GMM(size=C,d=d))
                X,Y=x[cv.iT[i],:], y[cv.iT[i]]
                nu = float(Y.size)
                for j in range(C):                  #Update the model
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
            while(r<maxvar):
                err=sp.zeros(variable.size)
                for i in range(v):                      # For each fold computes the prediction accuracy                    
                    partial_f=partial(compute_v_cv_gmm,model_cv=model_pre_cv[i],xt = x[cv.iT[i],:],yt=y[cv.iT[i]],ids=ids)
                    p = mp.Pool(processes=ncpus)
                    err += sp.asarray(p.map(partial_f,variable)) 
                    p.close()
                    p.join()

                err /= v
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
            
            
        ## Update the final model
        ids_s = ids[:]
        ids_s.sort()
        model_u = self.update(ids_s)
        del ids_s
        
        ## Return the final value
        return ids,OA,model_u  
