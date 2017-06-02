source("MultiTaskGradBoost.R")
library(MASS)
library(grpreg)
library(xtable)
library(knitr)
source("helper.R")
### our boolean functino is, for example: x1 and X2 and X7 and X8 (or some other function, maybe less linear)
### we classify as 1 all cases when this is TRUE and 0 otherwise
### each family has a slightly different probability of Xi being 1. 
boolfunc = function(x){
  

  i=x[length(x)]
  #ret = (xor(x[1],x[6])|xor(x[2],x[7])|xor(x[3],x[8])|xor(x[4],x[9])|xor(x[5],x[11])) & (xor(x[10],x[i]))
  ret = (xor(x[1],x[6])|xor(x[2],x[7])|xor(x[3],x[8]))&(xor(x[4],x[9])|xor(x[5],x[11]) | xor(x[10],x[i]))
  #ret = (xor(x[1],x[6])|xor(x[2],x[7])|xor(x[3],x[8])|xor(x[4],x[9])|xor(x[5],x[10])) & (x[10]&x[i])

  
  #return(ret)
  if(ret==0){
    return(-1)
  }else{
    return(1)
  }
}

# ## X has a "family" column
# ## we need to also add an intercept to each problem, as the grplaso package requests that in the design matrix
# ##
# CreateGroupLassoDesignMatrix = function(X){
#   
#   families = unique(X[,"Family"])
#   ntasks=length(families)
#   colsPerTask = (ncol(X)-2) # removing label and family columns
#   colsPerTask = colsPerTask+1 # adding intercept column per each task (two different rows of code just for readibility)
#   Xgl = matrix(0,nrow=nrow(X),ncol=(colsPerTask*ntasks)+1) # the +1 is for the label
#   cat(nrow(Xgl),"XGL",ncol(Xgl),"\n")
#   i=0
#   cat("ncol of xgl is:",ncol(Xgl),"\n")
#   groups=rep(1:colsPerTask,ntasks) # which group each variable belongs to. in our case, we group the same variable across tasks
#   for(fam in families){
#     cat(fam,"\n")
#     nr = nrow(X[X[,"Family"]==fam,]) ## how many instances for this family/task
#     nc = colsPerTask ## we know in our formulation that all tasks share the same number of features
#     rowstart=(i*nr)+1
#     rowend = (i+1)*nr
#     colstart = (i*nc)+1
#     colend = (i+1)*nc
#     cat(rowstart,"->",rowend,"\n")
#     cat(colstart,"->",colend,"\n")
#     Xgl[rowstart:rowend,colstart:(colend-1)]=as.matrix(X[X[,"Family"]==fam,-which(colnames(X) %in% c("Family","Label"))])
#     Xgl[rowstart:rowend,colend]=matrix(1,nrow=length(rowstart:rowend),ncol=1) ## add intercept per task
#     ## add label
#     Xgl[rowstart:rowend,ncol(Xgl)] = as.matrix(X[X[,"Family"]==fam,"Label"])
#     i=i+1
#     
#   }
#   ret=list()
#   ret[["X"]]=Xgl
#   ret[["groups"]]=groups
#   return(ret)
# }

GenerateNonLinearData=function(ntasks=5,d=50,ntrain=1000,ntest=300,seed=777){
  
  ## for each task, we generate a perturbation vector. this changes the probability of each bernoulli variable
  ## d bernulli variables overall with some independent distribution each to turn on/off
  ## each "family" is determined by whether some subset of flags was on
  ## if so it belogns to the fmaily, otherwise it's not
  ## 
  n=ntrain+ntest
  set.seed(seed)
  relevantdim = 11
  
  qs = runif(d,0.1,0.4) ## probability for each flag to be on
  #qs[c(1,2,3,4,5)]=runif(5,0.4,0.6)
  qs[c(1,2,3,4,5)]=runif(5,0.4,0.6)
  allmatrix=c()
  groups=c()
  
  ## make sure we stay within (0,1)
  
  #tq=c(0.03741054 ,0.85404431 ,0.46901895 ,0.23391561 ,0.10628011 ,0.57462687 ,0.45676139, 0.20296465, 0.13046421, 0.31055056)
  flagsmatrix=matrix(nrow=5,ncol=d)
  for(t in 1:ntasks){
    ### generate a sample of n=ntrain+ntest samples, each of dimension d
    ##generate a boolean vector using binomial distribution
    ## build the matrix for this task colum wise, we generate each column independently
    per = rnorm(n=d,mean=0,sd=0.07) ## perturbation per task, intuitively: the more we perturb, the less "in common" the tasks have
    tq=qs+per ## perturb probabilities for this task
    minval = min(tq[tq>0])
    maxval=max(tq[tq<1])
    tq[tq<=0]=minval
    tq[tq>=1]=maxval
    flagsmatrix[t,]=tq
    taskmatrix=matrix(nrow=n,ncol=relevantdim)
    for(dd in 1:relevantdim){
      
      p=tq[dd] ## the relevant probability for this column for this task
      taskmatrix[,dd] =rbinom(n,1,p) ## generate n samlpes with probability p
    }
    
    perturbedvars=d-relevantdim
    
    perturbmat = matrix( rnorm(perturbedvars*nrow(taskmatrix),mean=0,sd=1), nrow=nrow(taskmatrix)) 
    #perturbmat = cbind(matrix(0,nrow=nrow(taskmatrix),ncol=ncol(taskmatrix)-perturbedvars),perturbmat)
    #taskmatrix = taskmatrix+perturbmat ## add some noise 
    taskmatrix = cbind(taskmatrix,perturbmat) ## add many noise variables
    taskmatrix=cbind(taskmatrix,matrix(t,nrow=nrow(taskmatrix),ncol=1))
    allmatrix=rbind(allmatrix,taskmatrix)
    groups = rbind(groups,matrix(paste0("fam",t),nrow=nrow(taskmatrix),ncol=1))
  }
  y=apply(allmatrix,1,boolfunc)
  allmatrix=allmatrix[,-ncol(allmatrix)]
  df=data.frame(allmatrix)
  ret=list()
  df["Label"]=y
  df["Family"]=groups  
  ret[["data"]]=df
  ret[["groups"]]=groups
  # testidx = c()
  # for(fam in unique(df[,"Family"])){
  #   testidx = c(testidx, sample(which(df[,"Family"]==fam),ntest))
  # }
  
  testidx = c()
  validx = c()
  trainidx = c()
  for(fam in unique(df[,"Family"])){
    
    testidx = c(testidx, sample(which(df[,"Family"]==fam),floor(length(which(df[,"Family"]==fam))*0.4) ))
    famtrain = setdiff(which(df[,"Family"]==fam),testidx)
    famvalidx = sample(famtrain,0.1*length(famtrain) ) #val 10% of train
    validx = c(validx,famvalidx)
    famtrain = setdiff(famtrain,validx) # remove validation from train
    trainidx = c(trainidx,famtrain)
  }
  
  ret[["testidx"]]=testidx
  ret[["trainidx"]]=trainidx
  ret[["validx"]]=validx
  ret[["qs"]]=qs
  ret[["per"]]=per
  ret[["tq"]]=tq
  ret[["flagsmatrix"]]=flagsmatrix
  ### do the same output as GenerateData to have everything run smoothly
  return(ret)    
}





GenerateNonLinearData2 = function(ntasks=5, d=5,ntrain=1000,ntest=300,seed=300){
  #ntasks=5 ### we generate a row for each task, so for each task we have a set of parameter weights
  #d=15 ### how many dimensions we have. we have 5 dimensions with actual value, and 5 which will be noise
  #n=100 #samples per task
  n=ntrain+ntest
  set.seed(seed)
  mu=rep(0,7)
  Sigma=diag(c(0.5, 0.25, 0.1, 0.05, 0.15, 0.1, 0.15))
  
  W = mvrnorm(ntasks,mu,Sigma)  ### this generate a 5x5 matrix, a row of coefficients per task
  
  colnames(W)=NULL
  #zz = matrix(0,ntasks,d-5)
  #W = cbind(W,zz)
  
  
  ## each row of W is the "controller" (Wt) of each task
  
  ## now generate the random data X ~ uniform(0,1)^d
  
  M = c()
  Y = c()
  groups = c()
  for(i in 1:ntasks){
    X = matrix(runif(n*d),nrow=n,ncol=d)
    #offsets = c(1,1,1,1,1)
    #w = t(W)[,i] + c(offsets,rep(0,d-5))
    w = t(W)[,i]
    
    y = t(apply(X,1,function(x){c(x[1]^2,x[4]^2,x[1]*x[2],x[3]*x[5],x[2],x[4],1)})) %*% w  #f transform each row of X non linearly
    y = y + rnorm(n=length(y),mean=0,sd=0.0)
    class1idx=which(y>median(y))
    class0idx=which(y<=median(y))
    y[class1idx]=1
    y[class0idx]=-1
    M = rbind(M,X)
    Y = rbind(Y,y)
    groups = rbind(groups,matrix(paste0("fam",i),nrow=length(y),ncol=1))
  }
  ### clean = less than median
  ### mal = more than median
  df=data.frame(M)
  df["Label"]=Y
  df["Family"]=groups
  ## create test indexes
  # testidx = c()
  # for(fam in unique(df[,"Family"])){
  #   testidx = c(testidx, sample(which(df[,"Family"]==fam),ntest))
  # }
  
  for(fam in unique(df[,"Family"])){
    
    testidx = c(testidx, sample(which(df[,"Family"]==fam),floor(length(which(df[,"Family"]==fam))*0.4) ))
    famtrain = setdiff(which(df[,"Family"]==fam),testidx)
    famvalidx = sample(famtrain,0.1*length(famtrain) ) #val 10% of train
    validx = c(validx,famvalidx)
    famtrain = setdiff(famtrain,validx) # remove validation from train
    trainidx = c(trainidx,famtrain)
  }

    
  ret=list()
  ret[["data"]]=df
  ret[["groups"]]=groups
  ret[["W"]]=W
  ret[["testidx"]]=testidx
  
  return(ret)
}




GenerateLinearData = function(ntasks=5, d=50,ntrain=1000,ntest=300,seed=300){
  #ntasks=5 ### we generate a row for each task, so for each task we have a set of parameter weights
  #d=15 ### how many dimensions we have. we have 5 dimensions with actual value, and 5 which will be noise
  #n=100 #samples per task
  n=ntrain+ntest
  set.seed(seed)
  mu=rep(0,5)

  Sigma=diag(c(1,0.9,0.8,0.7,0.6)*0.5)
  W = mvrnorm(ntasks,mu,Sigma)  ### this generate a 5x5 matrix, a row of coefficients per task
  
  colnames(W)=NULL
  zz = matrix(0,ntasks,d-5)
  W = cbind(W,zz)
  
  
  ## each row of W is the "controller" (Wt) of each task
  
  ## now generate the random data X ~ uniform(0,1)^d
  ret=list()
  M = c()
  Y = c()
  groups = c()
  for(i in 1:ntasks){
    X = matrix(runif(n*d),nrow=n,ncol=d)
    offsets = c(1,1,1,1,1)
    w = t(W)[,i] + c(offsets,rep(0,d-5))
    y = X %*% w
    y = y + rnorm(n=length(y),mean=0,sd=0.0)
    ret[["rawscore"]]=y
    #y = 1/(1+exp(-0.5*y))
    # transform y to 0,1 uniformly
    #y = (y-min(y)) * (0.999/(max(y)-min(y)))+0.001 ### newValue = ((oldValue - oldMin) * newRange / oldRange) + newMin --> transform y to 0,1
    a=5
    y=(a*y)-median(a*y)
    #y=1/(1+exp(y))
    y=exp(y)/(1+exp(y))
    ret[["transformedscore"]]=y
    # perform logit
    
    

    # now we have a probability for each y to be in class 1 or 0
    
    y=rbinom(length(y),1,y) ## for each value of y, we took it as  aprobability to be in 1 or 0
    y[y==0]=-1 ## label 0 as -1
    ret[["y"]]=y
    #class1idx=which(y>0.6)
    #class0idx=which(y<=0.6)
    #y[class1idx]=1
    #y[class0idx]=-1
    #y[y==0]=-1
    
    M = rbind(M,X)
    Y = rbind(Y,matrix(y,nrow=length(y),ncol=1))
    groups = rbind(groups,matrix(paste0("fam",i),nrow=length(y),ncol=1))
  }
  
  ### clean = less than median
  ### mal = more than median
  df=data.frame(M)
  
  df["Label"]=Y
  
  df["Family"]=groups
  ## create test indexes
  testidx = c()
  for(fam in unique(df[,"Family"])){
    testidx = c(testidx, sample(which(df[,"Family"]==fam),ntest))
  }
  
  ret[["data"]]=df
  ret[["groups"]]=groups
  ret[["W"]]=W
  ret[["testidx"]]=testidx
  
  return(ret)
}

####### NON LINEAR ######

d=60
ntasks=5
ntest=10000
ntrain=1000
#controls=c(maxdepth=2,minbucket=1)
controls=rpart.control()
iter=150
rate=0.1
ridge.lambda=1  
#data=GenerateData(d=d,ntrain=ntrain,ntest=ntest,seed=i)
data=GenerateNonLinearData(d=d,ntasks=ntasks,ntrain=ntrain,ntest=ntest,seed=201)
train = data$data[data$trainidx,]
test = data$data[data$testidx,]
val = data$data[data$validx,]
mshared=TrainMultiTaskClassificationGradBoost(train,iter=iter,v=rate,valdata=val,groups=train[,"Family"],controls=controls,ridge.lambda=ridge.lambda,target="binary")
perTaskModels=list()
logitModels=list()
for(fam in unique(train[,"Family"])){
  
  tr = train[train[,"Family"]==fam,]
  tr.val = val[val[,"Family"]==fam,]
  
  m0 = TrainMultiTaskClassificationGradBoost(tr,valdata=val,groups = matrix(fam,nrow=nrow(tr),ncol=1),iter=iter,v=rate,
                                             controls=controls, ridge.lambda = ridge.lambda)  
  perTaskModels[[toString(fam)]]=m0
  logitModels[[toString(fam)]]= cv.glmnet(x=as.matrix(tr[,-which(colnames(tr) %in% c("Family","Label"))]),y=tr[,"Label"],family="binomial",alpha=1,maxit=10000,nfolds=4, thresh=1E-4)
}

### train binary model, ignoring multi tasking:
binaryData = train
binaryData["Family"]="1"
binaryVal = val
binaryVal["Family"]="1"
mbinary=TrainMultiTaskClassificationGradBoost(binaryData,valdata=binaryVal,iter=iter,v=rate,groups=matrix(1,nrow=nrow(binaryData),ncol=1),controls=controls,ridge.lambda=ridge.lambda)  
mlogitbinary = cv.glmnet(x=as.matrix(binaryData[,-which(colnames(tr) %in% c("Family","Label"))]),y=binaryData[,"Label"],family="binomial",alpha=1,maxit=10000,nfolds=4, thresh=1E-4)

gplassotraindata = CreateGroupLassoDesignMatrix(train)
gplassoX = (gplassotraindata$X)[,-ncol(gplassotraindata$X)]
gplassoy =  (gplassotraindata$X)[,ncol(gplassotraindata$X)]
gplassoy[gplassoy==-1]=0

mgplasso = cv.grpreg(gplassoX, gplassoy, group=gplassotraindata$groups, nfolds=5, seed=777,family="binomial",trace=TRUE)

gplassotestdata = CreateGroupLassoDesignMatrix(test)
gplassotestX = (gplassotestdata$X)[,-ncol(gplassotestdata$X)]
gplassotesty =  (gplassotestdata$X)[,ncol(gplassotestdata$X)]

gplassoPreds = predict(mgplasso,gplassotestX,type="response",lambda=mgplasso$lambda.min)   
methods = c("PANDO","PTB","BB","PTLogit","BinaryLogit","GL")
rc=list()
tt=list()
compmat = c()
digitsfmt = matrix(-2,nrow=length(methods),ncol=length(methods))
xtables=list()
k=0
##################### test:
for(fam in unique(test[,"Family"])){
  k = k+1
  compmatrix = matrix(nrow=length(methods),ncol = length(methods))
  #tr.test = test[test[,"Family"] %in% c(fam,"clean"),-which(colnames(test)=="Family")]
  tr.test = test[test["Family"]==fam,]
  tr.test = tr.test[,-which(colnames(tr.test)=="Family")]
  tt[[methods[1]]]=predict.Boost(tr.test[,-which(colnames(tr.test) %in% c("Family","Label"))],mshared[[toString(fam)]],rate=rate)
  rc[[methods[1]]] = pROC::roc(as.factor(tr.test[,"Label"]),tt[[methods[1]]])
  tt[[methods[2]]]=predict.Boost(tr.test[,-which(colnames(tr.test) %in% c("Family","Label"))],perTaskModels[[toString(fam)]][[toString(fam)]],rate=rate)
  rc[[methods[2]]] = pROC::roc(as.factor(tr.test[,"Label"]),tt[[methods[2]]])
  tt[[methods[3]]] = predict.Boost(tr.test[,-which(colnames(tr.test) %in% c("Family","Label"))],mbinary[[toString(1)]],rate=rate)
  rc[[methods[3]]] = pROC::roc(as.factor(tr.test[,"Label"]),tt[[methods[3]]])
  tt[[methods[4]]] =predict(logitModels[[fam]],newx=as.matrix(tr.test[,-which(colnames(tr) %in% c("Family","Label"))]),type="response",s=logitModels[[fam]]$lambda.min)
  rc[[methods[4]]] = pROC::roc(as.factor(tr.test[,"Label"]),as.numeric(tt[[methods[4]]]))
  tt[[methods[5]]] =predict(mlogitbinary,newx=as.matrix(tr.test[,-which(colnames(tr) %in% c("Family","Label"))]),type="response",s=mlogitbinary$lambda.min)
  rc[[methods[5]]] = pROC::roc(as.factor(tr.test[,"Label"]),as.numeric(tt[[methods[5]]]))
  tt[[methods[6]]] = gplassoPreds[test[,"Family"]==fam]
  rc[[methods[6]]] = pROC::roc(as.factor(tr.test[,"Label"]),as.numeric(tt[[methods[6]]]))
  
  for(i in 1:6){
    #cat("storing ",round(pROC::auc(rc[[methods[i]]])[1],4)," in ",i,i,"\n" )
    compmatrix[i,i]=round(pROC::auc(rc[[methods[i]]])[1],4) # store auc of this method
    digitsfmt[i,i]=3
    for(j in 1:6){
      if(i >=j ){
        next
      }
      #cat("storing ",signif(pROC::roc.test(rc[[methods[i]]],rc[[methods[j]]])$p.value, digits = 3)," now in ",i,j,"\n")
      compmatrix[i,j] = signif(pROC::roc.test(rc[[methods[i]]],rc[[methods[j]]])$p.value, digits = 3)
      cat("AUROC for ",fam," ", methods[i]," VS ",methods[j],": ",round(pROC::auc(rc[[methods[i]]])[1],4)," ",round(pROC::auc(rc[[methods[j]]])[1],4),"diff p-val:",signif(pROC::roc.test(rc[[methods[i]]],rc[[methods[j]]])$p.value, digits = 3),"\n")
    }
  }
  compmat = rbind(compmat,compmatrix)
  
#  cat("AUROC for ",fam," PANDO VS per-task boosted trees is: ",pROC::auc(rc1)[1]," ",pROC::auc(rc2)[1],"diff p-val:",signif(pROC::roc.test(rc1,rc2)$p.value, digits = 3),"\n")
#  cat("AUROC for ",fam," PANDO VS BinaryBoosting boosted trees is: ",pROC::auc(rc1)[1]," ",pROC::auc(rc3)[1],"diff p-val:",signif(pROC::roc.test(rc1,rc3)$p.value, digits = 3),"\n")
#  cat("AUROC for ",fam," PANDO VS Per Task logistic regression is: ",pROC::auc(rc1)[1]," ",pROC::auc(rc4)[1],"diff p-val:",signif(pROC::roc.test(rc1,rc4)$p.value, digits = 3),"\n")
#  cat("AUROC for ",fam," PANDO VS Binary Logistic Regression is: ",pROC::auc(rc1)[1]," ",pROC::auc(rc5)[1],"diff p-val:",signif(pROC::roc.test(rc1,rc5)$p.value, digits = 3),"\n")
#  cat("AUROC for ",fam," PerTaskBoosting VS BinaryBoosting: ",pROC::auc(rc2)[1]," ",pROC::auc(rc3)[1],"diff p-val:",signif(pROC::roc.test(rc2,rc3)$p.value, digits = 3),"\n")
#  cat("AUROC for ",fam," PerTaskBoosting VS PTLogit: ",pROC::auc(rc2)[1]," ",pROC::auc(rc4)[1],"diff p-val:",signif(pROC::roc.test(rc2,rc4)$p.value, digits = 3),"\n")
#  cat("AUROC for ",fam," PANDO VS gplasso: ",pROC::auc(rc1)[1]," ",pROC::auc(rc6)[1],"diff p-val:",signif(pROC::roc.test(rc1,rc6)$p.value, digits = 3),"\n")
#  cat("AUROC for ",fam," Per Task logistic regression VS gplasso: ",pROC::auc(rc4)[1]," ",pROC::auc(rc6)[1],"diff p-val:",signif(pROC::roc.test(rc4,rc6)$p.value, digits = 3),"\n")
  cat("***********\n")
  
  
  dft=data.frame(compmatrix)
  colnames(dft)=methods
  rownames(dft)=methods
  xtables[[toString(k)]]=xtable(dft,digits=cbind(rep(1,nrow(digitsfmt)),digitsfmt))
  
}
####################################################################################################
  


###### LINEAR ########

d=25
ntest=10000
ntrain=1000
controls=rpart.control()
iter=200
rate=0.01
ridge.lambda=1  
data=GenerateLinearData(d=d,ntrain=ntrain,ntest=ntest)
train = data$data[-data$testidx,]
test = data$data[data$testidx,]
mshared=TrainMultiTaskClassificationGradBoost(train,iter=iter,v=rate,groups=train[,"Family"],controls=controls,ridge.lambda=ridge.lambda)
perTaskModels=list()
logitModels=list()
for(fam in unique(train[,"Family"])){
  cat("wow**********************************\n")
  tr = train[train[,"Family"]==fam,]
  m0 = TrainMultiTaskClassificationGradBoost(tr,groups = matrix(fam,nrow=nrow(tr),ncol=1),iter=iter,v=rate,
                                             controls=controls, ridge.lambda = ridge.lambda)  
  perTaskModels[[toString(fam)]]=m0
  logitModels[[toString(fam)]]= cv.glmnet(x=as.matrix(tr[,-which(colnames(tr) %in% c("Family","Label"))]),y=tr[,"Label"],family="binomial",alpha=1,maxit=10000,nfolds=4, thresh=1E-4)
}

### train binary model, ignoring multi tasking:
binaryData = train
binaryData["Family"]=1
mbinary=TrainMultiTaskClassificationGradBoost(binaryData,iter=iter,v=rate,groups=matrix(1,nrow=nrow(binaryData),ncol=1),controls=controls,ridge.lambda=ridge.lambda)
mlogitbinary = cv.glmnet(x=as.matrix(binaryData[,-which(colnames(tr) %in% c("Family","Label"))]),y=binaryData[,"Label"],family="binomial",alpha=1,maxit=10000,nfolds=4, thresh=1E-4)

gplassotraindata = CreateGroupLassoDesignMatrix(train)
gplassoX = (gplassotraindata$X)[,-ncol(gplassotraindata$X)]
gplassoy =  (gplassotraindata$X)[,ncol(gplassotraindata$X)]
gplassoy[gplassoy==-1]=0

mgplasso = cv.grpreg(gplassoX, gplassoy, group=gplassotraindata$groups, nfolds=5, seed=777,family="binomial",trace=TRUE)

gplassotestdata = CreateGroupLassoDesignMatrix(test)
gplassotestX = (gplassotestdata$X)[,-ncol(gplassotestdata$X)]
gplassotesty =  (gplassotestdata$X)[,ncol(gplassotestdata$X)]

gplassoPreds = predict(mgplasso,gplassotestX,type="response",lambda=mgplasso$lambda.min)    

methods = c("PANDO","PTB","BB","PTLogit","BinaryLogit","GL")
compmat = c()
digitsfmt = matrix(-2,nrow=length(methods),ncol=length(methods))
xtables=list()
tt=list()
rc=list()
k=0
##################### test:
for(fam in unique(test[,"Family"])){
  k=k+1
  compmatrix = matrix(nrow=length(methods),ncol = length(methods))
  #tr.test = test[test[,"Family"] %in% c(fam,"clean"),-which(colnames(test)=="Family")]
  tr.test = test[test["Family"]==fam,]
  tr.test = tr.test[,-which(colnames(tr.test)=="Family")]

  tt[[methods[1]]]=predict.Boost(tr.test[,-which(colnames(tr.test) %in% c("Family","Label"))],mshared[[toString(fam)]],rate=rate)
  rc[[methods[1]]] = pROC::roc(as.factor(tr.test[,"Label"]),tt[[methods[1]]])
  tt[[methods[2]]]=predict.Boost(tr.test[,-which(colnames(tr.test) %in% c("Family","Label"))],perTaskModels[[toString(fam)]][[toString(fam)]],rate=rate)
  rc[[methods[2]]] = pROC::roc(as.factor(tr.test[,"Label"]),tt[[methods[2]]])
  tt[[methods[3]]] = predict.Boost(tr.test[,-which(colnames(tr.test) %in% c("Family","Label"))],mbinary[[toString(1)]],rate=rate)
  rc[[methods[3]]] = pROC::roc(as.factor(tr.test[,"Label"]),tt[[methods[3]]])
  tt[[methods[4]]] =predict(logitModels[[fam]],newx=as.matrix(tr.test[,-which(colnames(tr) %in% c("Family","Label"))]),type="response",s=logitModels[[fam]]$lambda.min)
  rc[[methods[4]]] = pROC::roc(as.factor(tr.test[,"Label"]),as.numeric(tt[[methods[4]]]))
  tt[[methods[5]]] =predict(mlogitbinary,newx=as.matrix(tr.test[,-which(colnames(tr) %in% c("Family","Label"))]),type="response",s=mlogitbinary$lambda.min)
  rc[[methods[5]]] = pROC::roc(as.factor(tr.test[,"Label"]),as.numeric(tt[[methods[5]]]))
  tt[[methods[6]]] = gplassoPreds[test[,"Family"]==fam]
  rc[[methods[6]]] = pROC::roc(as.factor(tr.test[,"Label"]),as.numeric(tt[[methods[6]]]))
  
  
  for(i in 1:6){
    #cat("storing ",round(pROC::auc(rc[[methods[i]]])[1],4)," in ",i,i,"\n" )
    compmatrix[i,i]=round(pROC::auc(rc[[methods[i]]])[1],4) # store auc of this method
    digitsfmt[i,i]=3
    for(j in 1:6){
      if(i >=j ){
        next
      }
      #cat("storing ",signif(pROC::roc.test(rc[[methods[i]]],rc[[methods[j]]])$p.value, digits = 3)," now in ",i,j,"\n")
      compmatrix[i,j] = signif(pROC::roc.test(rc[[methods[i]]],rc[[methods[j]]])$p.value, digits = 3)
      cat("AUROC for ",fam," ", methods[i]," VS ",methods[j],": ",round(pROC::auc(rc[[methods[i]]])[1],4)," ",round(pROC::auc(rc[[methods[j]]])[1],4),"diff p-val:",signif(pROC::roc.test(rc[[methods[i]]],rc[[methods[j]]])$p.value, digits = 3),"\n")
    }
  }
  compmat = rbind(compmat,compmatrix)
  
  dft=data.frame(compmatrix)
  colnames(dft)=methods
  rownames(dft)=methods
  xtables[[toString(k)]]=xtable(dft,digits=cbind(rep(1,nrow(digitsfmt)),digitsfmt))
  
  #cat("AUROC for ",fam," VS per-task boosted trees is: ",pROC::auc(rc1)[1]," ",pROC::auc(rc2)[1],"diff p-val:",pROC::roc.test(rc1,rc2)$p.value,"\n")
  #cat("AUROC for ",fam," VS malware-clean boosted trees is: ",pROC::auc(rc1)[1]," ",pROC::auc(rc3)[1],"diff p-val:",pROC::roc.test(rc1,rc3)$p.value,"\n")
  #cat("AUROC for ",fam," VS PerTaskLogit is: ",pROC::auc(rc1)[1]," ",pROC::auc(rc4)[1],"diff p-val:",pROC::roc.test(rc1,rc4)$p.value,"\n")
  #cat("AUROC for ",fam," PerTaskBoosting VS BinaryBoosting: ",pROC::auc(rc2)[1]," ",pROC::auc(rc3)[1],"diff p-val:",pROC::roc.test(rc2,rc3)$p.value,"\n")
  #cat("AUROC for ",fam," PerTaskBoosting VS gplasso: ",pROC::auc(rc2)[1]," ",pROC::auc(rc6)[1],"diff p-val:",pROC::roc.test(rc2,rc6)$p.value,"\n")
  #cat("AUROC for ",fam," BinaryLogit VS gplasso: ",pROC::auc(rc5)[1]," ",pROC::auc(rc6)[1],"diff p-val:",pROC::roc.test(rc5,rc6)$p.value,"\n")
  #cat("AUROC for ",fam," PerTaskLogit VS gplasso: ",pROC::auc(rc4)[1]," ",pROC::auc(rc6)[1],"diff p-val:",pROC::roc.test(rc4,rc6)$p.value,"\n")
  #cat("AUROC for ",fam," PANDO VS gplasso: ",pROC::auc(rc1)[1]," ",pROC::auc(rc6)[1],"diff p-val:",pROC::roc.test(rc1,rc6)$p.value,"\n")
  cat("***********\n")
}
####################################################################################################


















plotImp2 = function(df,signalVars=c(), title){
  
  df[,"value"]=df[,"value"]/sum(df[,"value"])
  colnames(df)=c("varname","value")
  #df=df[order(-df[,"value"]),]
  df <- base::transform(df, varname = reorder(varname, -value))
  df[,"Legend"]="noise"
  df[df[,"varname"] %in% signalVars,"Legend"]="signal"
  group.colors <- c(noise = "#C67171", signal = "#7171C6") 
  p3 = ggplot(head(df,n=40), aes(varname, weight = value,fill=Legend)) + geom_bar()
  p3 = p3+scale_fill_manual(values=group.colors)
  p3 = p3 + labs(title = title)
  p3 = p3+ylab("split gain")
  
  p3=p3+thm
  #p3=p3+coord_flip()
  p3  
  
}


dff <- melt(ldply(imp1, data.frame))
dff=dff[,-2]
colnames(dff)=c("varname","value")
dff=dff[order(-dff[,"value"]),]
dffnorm=dff
dffnorm[,"value"]=dffnorm[,"value"]/sum(dffnorm[,"value"])
colnames(dffnorm)=c("varname","value")
#dffnorm=dffnorm[order(-dffnorm[,"value"]),]
dffnorm <- base::transform(dffnorm, varname = reorder(varname, -value))
#plotImp(dffnorm,paste0("X",1:5),"cumm. variable importance across tasks - PANDO")
plotImp(dffnorm,paste0("X",1:11),"cumm. variable importance across tasks - PANDO")


dff2 <- melt(ldply(imp2, data.frame))
dff2=dff2[,-2]
colnames(dff2)=c("varname","value")
dff2=dff2[order(-dff2[,"value"]),]
dff2norm=dff2
dff2norm[,"value"]=dff2norm[,"value"]/sum(dff2norm[,"value"])
colnames(dff2norm)=c("varname","value")
#dff2norm=dff2norm[order(-dff2norm[,"value"]),]
dff2norm <- base::transform(dff2norm, varname = reorder(varname, -value))
#plotImp(dff2norm,paste0("X",1:5),"cumm. variable importance across tasks - PTB")
plotImp(dff2norm,paste0("X",1:11),"cumm. variable importance across tasks - PTB")


dff3 <- melt(ldply(imp3, data.frame))
dff3=dff3[,-2]
colnames(dff3)=c("varname","value")
dff3=dff3[order(-dff3[,"value"]),]
dff3norm=dff3
dff3norm[,"value"]=dff3norm[,"value"]/sum(dff3norm[,"value"])
colnames(dff3norm)=c("varname","value")
#dff3norm=dff3norm[order(-dff3norm[,"value"]),]
dff3norm <- base::transform(dff3norm, varname = reorder(varname, -value))
#plotImp(dff3norm,paste0("X",1:5),"cumm. variable importance across tasks - BinaryBoosting")
plotImp(dff3norm,paste0("X",1:11),"cumm. variable importance across tasks - BinaryBoosting")




### normalized versions:
dffnorm=dff
dffnorm[,"value"]=dffnorm[,"value"]/sum(dffnorm[,"value"])
colnames(dffnorm)=c("varname","value")
#dffnorm=dffnorm[order(-dffnorm[,"value"]),]
dffnorm <- base::transform(dffnorm, varname = reorder(varname, -value))
dffnorm[,"Legend"]="noise"
dffnorm[dffnorm[,"varname"] %in% c(paste0("X",1:5)),"Legend"]="signal"
group.colors <- c(noise = "#C67171", signal = "#7171C6") 
p3 = ggplot(head(dffnorm,n=20), aes(varname, weight = value,fill=Legend)) + geom_bar()
p3 = p3+scale_fill_manual(values=group.colors)
p3 = p3 + labs(title = paste("cumm. normalized variable importance across tasks - PANDO"))
p3 = p3+ylab("split gain")

p3=p3+thm
p3
rpart.plot(mshared$fam1[[2]]$treeFit)  ## displaying a tree by mshared

### normalized versions:

dff2norm=dff2
dff2norm[,"value"]=dff2norm[,"value"]/sum(dff2norm[,"value"])
colnames(dff2norm)=c("varname","value")
#dff2norm=dff2norm[order(-dff2norm[,"value"]),]
dff2norm <- base::transform(dff2norm, varname = reorder(varname, -value))
dff2norm[,"Legend"]="noise"
dff2norm[dff2norm[,"varname"] %in% c(paste0("X",1:5)),"Legend"]="signal"
group.colors <- c(noise = "#C67171", signal = "#7171C6") 
p4 = ggplot(head(dff2norm,n=20), aes(varname, weight = value,fill=Legend)) + geom_bar()
p4 = p4+scale_fill_manual(values=group.colors)
p4 = p4 + labs(title = paste("cumm. normalized variable importance across tasks - PTB"))
p4 = p4+ylab("split gain")

p4=p4+thm
p4


dff3norm=dff3
dff3norm[,"value"]=dff3norm[,"value"]/sum(dff3norm[,"value"])
colnames(dff3norm)=c("varname","value")
#dff3norm=dff3norm[order(-dff3norm[,"value"]),]
dff3norm <- base::transform(dff3norm, varname = reorder(varname, -value))
dff3norm[,"color"]="#333BFF"
dff3norm[dff3norm[,"varname"] %in% c(paste0("X",1:5)),"color"]="#CC6600"

#p = ggplot(dff, aes(varname, weight = value,fill=color)) + geom_bar()
p5 = ggplot(head(dff3norm,n=20), aes(varname, weight = value,fill=color)) + geom_bar()
p5 = p5 + labs(title = paste("cumm. normalized variable importance across tasks - BB"))
p5 = p5+ylab("split gain")

p5=p5+thm
p5






dd = matrix(coef(mgplasso))
dd=matrix(dd)
dd=data.frame(matrix(dd[-1,]))
dd[,"varname"]=paste0(paste0("T",rep(1:5,each=26),"_"),rep(paste0("X",1:26),5))
colnames(dd)=c("value","varname")
dd <- dd[order(-abs(dd[,"value"])),]

pgp = ggplot(head(dd,n=40), aes(varname, weight = value)) + geom_bar()
pgp = pgp + labs(title = paste("var importance gplasso"))
pgp = pgp+ylab("linear coef")

pgp=pgp+thm
pgp + coord_flip()
pgp


library(rpart.plot)
bdata=binaryData[,-ncol(binaryData)]
v=.1 
fit=rpart(Label~.,data=bdata,controls=controls)
yp=predict(fit)
df$yr=df$y - v*yp
YP=v*yp
for(t in 1:100){
  fit=rpart(yr~x,data=df)
  yp=predict(fit,newdata=df)
  df$yr=df$yr - v*yp
  YP=cbind(YP,v*yp)
}









