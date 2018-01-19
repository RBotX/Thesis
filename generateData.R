source("PANDO.r")
source("PANDO2.r")
source("PandoObo.R")
source("helper.R")
library(MASS)
library(grpreg)
library(xtable)
library(knitr)
source("helper.R")
### our boolean functino is, for example: x1 and X2 and X7 and X8 (or some other function, maybe less linear)
### we classify as 1 all cases when this is TRUE and 0 otherwise
### each family has a slightly different probability of Xi being 1. 
outdir="PandoSimulationResults"
dir.create(outdir,showWarnings = FALSE)

boolfunc = function(x){
  

  i=x[length(x)]
  
  #ret = (xor(x[1],x[6])|xor(x[2],x[7])|xor(x[3],x[8]))&(xor(x[4],x[9])|xor(x[5],x[11]) | xor(x[10],x[i]))
  ret = (xor(x[1],x[6])|xor(x[2],x[7])|xor(x[3],x[i]))&(xor(x[4],x[9])|xor(x[5],x[11]) | xor(x[10],x[i]))
  

  
  #return(ret)
  if(ret==0){
    return(-1)
  }else{
    return(1)
  }
}


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
  #qs[1:ntasks]=runif(ntasks,0.3,0.7)
  qs[1:ntasks]=runif(ntasks,0.4,0.7)
  allmatrix=c()
  groups=c()
  
  ## make sure we stay within (0,1)
  
  #tq=c(0.03741054 ,0.85404431 ,0.46901895 ,0.23391561 ,0.10628011 ,0.57462687 ,0.45676139, 0.20296465, 0.13046421, 0.31055056)
  flagsmatrix=matrix(nrow=ntasks,ncol=d)
  for(t in 1:ntasks){
    ### generate a sample of n=ntrain+ntest samples, each of dimension d
    ##generate a boolean vector using binomial distribution
    ## build the matrix for this task colum wise, we generate each column independently
    #per = rnorm(n=d,mean=0,sd=0.07) ## perturbation per task, intuitively: the more we perturb, the less "in common" the tasks have
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
    
    testidx = c(testidx, sample(which(df[,"Family"]==fam),ntest ))
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
  validx = c()
  trainidx = c()
  for(fam in unique(df[,"Family"])){
    
    testidx = c(testidx, sample(which(df[,"Family"]==fam),ntest ))
    famtrain = setdiff(which(df[,"Family"]==fam),testidx)
    famvalidx = sample(famtrain,0.1*length(famtrain) ) #val 10% of train
    validx = c(validx,famvalidx)
    famtrain = setdiff(famtrain,validx) # remove validation from train
    trainidx = c(trainidx,famtrain)
  }
  
  ret[["testidx"]]=testidx
  ret[["trainidx"]]=trainidx
  ret[["validx"]]=validx
  
  ret[["data"]]=df
  ret[["groups"]]=groups
  ret[["W"]]=W
  
  
  return(ret)
}

####### NON LINEAR ######

d=150
ntasks=5
ntest=10000
ntrain=1000
#controls=c(maxdepth=2,minbucket=1)
controls=rpart.control()
iter=1000
rate=0.01
ridge.lambda=1  
#data=GenerateData(d=d,ntrain=ntrain,ntest=ntest,seed=i)
data=GenerateNonLinearData(d=d,ntasks=ntasks,ntrain=ntrain,ntest=ntest,seed=201)
train = data$data[data$trainidx,]
test = data$data[data$testidx,]
val = data$data[data$validx,]
mshared=TrainMultiTaskClassificationGradBoost(train,iter=iter,v=rate,valdata=val,groups=train[,"Family"],controls=controls,ridge.lambda=ridge.lambda,target="binary")
mshared2=TrainMultiTaskClassificationGradBoost2(train,iter=iter,v=rate,groups=train[,"Family"],controls=controls,ridge.lambda=ridge.lambda,target="binary",treeType="rpart",valdata=val)
perTaskModels=list()
logitModels=list()
for(fam in unique(train[,"Family"])){
  
  tr = train[train[,"Family"]==fam,]
  tr.val = val[val[,"Family"]==fam,]
  
  m0 = TrainMultiTaskClassificationGradBoost(tr,valdata=val,groups = matrix(fam,nrow=nrow(tr),ncol=1),iter=iter,v=rate,
                                             controls=controls, ridge.lambda = ridge.lambda,target="binary")  
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

mgplasso = cv.grpreg(gplassoX, gplassoy, group=gplassotraindata$groups, nfolds=2, seed=777,family="binomial",trace=TRUE)

gplassotestdata = CreateGroupLassoDesignMatrix(test)
gplassotestX = (gplassotestdata$X)[,-ncol(gplassotestdata$X)]
gplassotesty =  (gplassotestdata$X)[,ncol(gplassotestdata$X)]

gplassoPreds = predict(mgplasso,gplassotestX,type="response",lambda=mgplasso$lambda.min)   

methods = c("PANDO","PANDO2","PTB","BB","PTLogit","BinaryLogit","GL")
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
  
  bestIt=min(which(as.vector(mshared$log$vscore)==max(as.vector(mshared$log$vscore))))
  cat("pando1\n")
  tt[[methods[which(methods=="PANDO")]]]=predict(mshared[[toString(fam)]],tr.test[,-which(colnames(tr.test) %in% c("Family","Label"))],calibrate=TRUE,bestIt=bestIt)
  rc[[methods[which(methods=="PANDO")]]] = pROC::roc(as.factor(tr.test[,"Label"]),as.numeric(tt[[methods[which(methods=="PANDO")]]]))
  
  cat("pando2\n")
  bestIt=min(which(as.vector(mshared2$log$vscore)==max(as.vector(mshared2$log$vscore))))    
  tt[[methods[which(methods=="PANDO2")]]] = predict(mshared2[[toString(fam)]],tr.test[,-which(colnames(tr.test) %in% c("Family","Label"))],calibrate=TRUE,bestIt=bestIt)
  rc[[methods[which(methods=="PANDO2")]]] = pROC::roc(as.factor(tr.test[,"Label"]),as.numeric(tt[[methods[which(methods=="PANDO2")]]]))


  cat("ptb\n")
  bestIt=min(which(as.vector(perTaskModels[[toString(fam)]]$log$vscore)==max(as.vector(perTaskModels[[toString(fam)]]$log$vscore))))    
  tt[[methods[which(methods=="PTB")]]]=predict(perTaskModels[[toString(fam)]][[toString(fam)]],tr.test[,-which(colnames(tr.test) %in% c("Family","Label"))],calibrate=TRUE,bestIt=bestIt)
  rc[[methods[which(methods=="PTB")]]] = pROC::roc(as.factor(tr.test[,"Label"]),as.numeric(tt[[methods[which(methods=="PTB")]]]))
  
  cat("bb\n")
  bestIt=min(which(as.vector(mbinary$log$vscore)==max(as.vector(mbinary$log$vscore))))    
  tt[[methods[which(methods=="BB")]]] = predict(mbinary[[toString(1)]],tr.test[,-which(colnames(tr.test) %in% c("Family","Label"))],calibrate=TRUE,bestIt=bestIt)
  rc[[methods[which(methods=="BB")]]] = pROC::roc(as.factor(tr.test[,"Label"]),as.numeric(tt[[methods[which(methods=="BB")]]]))
  
  tt[[methods[which(methods=="PTLogit")]]] =predict(logitModels[[fam]],newx=as.matrix(tr.test[,-which(colnames(tr) %in% c("Family","Label"))]),type="response",s=logitModels[[fam]]$lambda.min)
  rc[[methods[which(methods=="PTLogit")]]] = pROC::roc(as.factor(tr.test[,"Label"]),as.numeric(tt[[methods[which(methods=="PTLogit")]]]))
  
  tt[[methods[which(methods=="BinaryLogit")]]] =predict(mlogitbinary,newx=as.matrix(tr.test[,-which(colnames(tr) %in% c("Family","Label"))]),type="response",s=mlogitbinary$lambda.min)
  rc[[methods[which(methods=="BinaryLogit")]]] = pROC::roc(as.factor(tr.test[,"Label"]),as.numeric(tt[[methods[which(methods=="BinaryLogit")]]]))
  tt[[methods[which(methods=="GL")]]] = gplassoPreds[test[,"Family"]==fam]
  rc[[methods[which(methods=="GL")]]] = pROC::roc(as.factor(tr.test[,"Label"]),as.numeric(tt[[methods[which(methods=="GL")]]]))
  
  
  
  for(i in 1:length(methods)){
    #cat("storing ",round(pROC::auc(rc[[methods[i]]])[1],4)," in ",i,i,"\n" )
    compmatrix[i,i]=round(pROC::auc(rc[[methods[i]]])[1],4) # store auc of this method
    digitsfmt[i,i]=3
    for(j in 1:length(methods)){
      if(i >=j ){
        next
      }
      #cat("storing ",signif(pROC::roc.test(rc[[methods[i]]],rc[[methods[j]]])$p.value, digits = 3)," now in ",i,j,"\n")
      compmatrix[i,j] = signif(pROC::roc.test(rc[[methods[i]]],rc[[methods[j]]])$p.value, digits = 3)
      cat("AUROC for ",fam," ", methods[i]," VS ",methods[j],": ",round(pROC::auc(rc[[methods[i]]])[1],4)," ",round(pROC::auc(rc[[methods[j]]])[1],4),"diff p-val:",signif(pROC::roc.test(rc[[methods[i]]],rc[[methods[j]]])$p.value, digits = 3),"\n")
    }
  }
  compmat = rbind(compmat,compmatrix)
  
  cat("***********\n")
  
  
  dft=data.frame(compmatrix)
  colnames(dft)=methods
  rownames(dft)=methods
  xtables[[toString(k)]]=xtable(dft,digits=cbind(rep(1,nrow(digitsfmt)),digitsfmt),caption=paste0("non linear task ",k))
  #print(xtables$`1`, table.placement="H")
}
####################################################################################################


#### plot feature importances and save locally:
plotExperimentResults(pandoModel = mshared,perTaskModels = perTaskModels,postfix = "NonLinear",signalVars = 1:11)

# dff = FeatureImportance.BoostingModel(mshared$fam1) ### for pando, we can use only one family as they share the same tree structures 
# PlotImp(dff,signalVars = 1:11,flip = TRUE)
# 
# dev.copy(png,filename=paste0(outdir,"/PandoFeatureImportanceNonLinear.png"))
# dev.off
# 
# 
# dff = PerTaskImportances(perTaskModels) ### for ptb
# PlotImp(dff,signalVars = 1:11,flip = TRUE)
# 
# dev.copy(png,filename=paste0(outdir,"/PTBFeatureImportanceNonLinear.png"))
# dev.off



###### LINEAR ########

d=25
ntest=10000
ntrain=1000
controls=rpart.control()
iter=200
rate=0.01
ridge.lambda=1  
data=GenerateLinearData(d=d,ntrain=ntrain,ntest=ntest)
train = data$data[data$trainidx,]
test = data$data[data$testidx,]
val = data$data[data$validx,]
mshared=TrainMultiTaskClassificationGradBoost(train,iter=iter,v=rate,valdata=val,groups=train[,"Family"],controls=controls,ridge.lambda=ridge.lambda,target="binary")
mshared2=TrainMultiTaskClassificationGradBoost2(train,iter=iter,v=rate,groups=train[,"Family"],controls=controls,ridge.lambda=ridge.lambda,target="binary",treeType="rpart",valdata=val)
perTaskModels=list()
logitModels=list()
for(fam in unique(train[,"Family"])){
  
  tr = train[train[,"Family"]==fam,]
  tr.val = val[val[,"Family"]==fam,]
  
  m0 = TrainMultiTaskClassificationGradBoost(tr,valdata=val,groups = matrix(fam,nrow=nrow(tr),ncol=1),iter=iter,v=rate,
                                             controls=controls, ridge.lambda = ridge.lambda,target="binary")  
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

mgplasso = cv.grpreg(gplassoX, gplassoy, group=gplassotraindata$groups, nfolds=2, seed=777,family="binomial",trace=TRUE)

gplassotestdata = CreateGroupLassoDesignMatrix(test)
gplassotestX = (gplassotestdata$X)[,-ncol(gplassotestdata$X)]
gplassotesty =  (gplassotestdata$X)[,ncol(gplassotestdata$X)]

gplassoPreds = predict(mgplasso,gplassotestX,type="response",lambda=mgplasso$lambda.min)   

methods = c("PANDO","PANDO2","PTB","BB","PTLogit","BinaryLogit","GL")
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
  
  bestIt=min(which(as.vector(mshared$log$vscore)==max(as.vector(mshared$log$vscore))))
  cat("pando1\n")
  tt[[methods[which(methods=="PANDO")]]]=predict(mshared[[toString(fam)]],tr.test[,-which(colnames(tr.test) %in% c("Family","Label"))],calibrate=TRUE,bestIt=bestIt)
  rc[[methods[which(methods=="PANDO")]]] = pROC::roc(as.factor(tr.test[,"Label"]),as.numeric(tt[[methods[which(methods=="PANDO")]]]))
  
  cat("pando2\n")
  bestIt=min(which(as.vector(mshared2$log$vscore)==max(as.vector(mshared2$log$vscore))))    
  tt[[methods[which(methods=="PANDO2")]]] = predict(mshared2[[toString(fam)]],tr.test[,-which(colnames(tr.test) %in% c("Family","Label"))],calibrate=TRUE,bestIt=bestIt)
  rc[[methods[which(methods=="PANDO2")]]] = pROC::roc(as.factor(tr.test[,"Label"]),as.numeric(tt[[methods[which(methods=="PANDO2")]]]))
  
  cat("ptb\n")
  bestIt=min(which(as.vector(perTaskModels[[toString(fam)]]$log$vscore)==max(as.vector(perTaskModels[[toString(fam)]]$log$vscore))))    
  tt[[methods[which(methods=="PTB")]]]=predict(perTaskModels[[toString(fam)]][[toString(fam)]],tr.test[,-which(colnames(tr.test) %in% c("Family","Label"))],calibrate=TRUE,bestIt=bestIt)
  rc[[methods[which(methods=="PTB")]]] = pROC::roc(as.factor(tr.test[,"Label"]),as.numeric(tt[[methods[which(methods=="PTB")]]]))
  
  cat("bb\n")
  bestIt=min(which(as.vector(mbinary$log$vscore)==max(as.vector(mbinary$log$vscore))))    
  tt[[methods[which(methods=="BB")]]] = predict(mbinary[[toString(1)]],tr.test[,-which(colnames(tr.test) %in% c("Family","Label"))],calibrate=TRUE,bestIt=bestIt)
  rc[[methods[which(methods=="BB")]]] = pROC::roc(as.factor(tr.test[,"Label"]),as.numeric(tt[[methods[which(methods=="BB")]]]))
  
  tt[[methods[which(methods=="PTLogit")]]] =predict(logitModels[[fam]],newx=as.matrix(tr.test[,-which(colnames(tr) %in% c("Family","Label"))]),type="response",s=logitModels[[fam]]$lambda.min)
  rc[[methods[which(methods=="PTLogit")]]] = pROC::roc(as.factor(tr.test[,"Label"]),as.numeric(tt[[methods[which(methods=="PTLogit")]]]))
  
  tt[[methods[which(methods=="BinaryLogit")]]] =predict(mlogitbinary,newx=as.matrix(tr.test[,-which(colnames(tr) %in% c("Family","Label"))]),type="response",s=mlogitbinary$lambda.min)
  rc[[methods[which(methods=="BinaryLogit")]]] = pROC::roc(as.factor(tr.test[,"Label"]),as.numeric(tt[[methods[which(methods=="BinaryLogit")]]]))
  tt[[methods[which(methods=="GL")]]] = gplassoPreds[test[,"Family"]==fam]
  rc[[methods[which(methods=="GL")]]] = pROC::roc(as.factor(tr.test[,"Label"]),as.numeric(tt[[methods[which(methods=="GL")]]]))
  
  
  
  for(i in 1:length(methods)){
    #cat("storing ",round(pROC::auc(rc[[methods[i]]])[1],4)," in ",i,i,"\n" )
    compmatrix[i,i]=round(pROC::auc(rc[[methods[i]]])[1],4) # store auc of this method
    digitsfmt[i,i]=3
    for(j in 1:length(methods)){
      if(i >=j ){
        next
      }
      #cat("storing ",signif(pROC::roc.test(rc[[methods[i]]],rc[[methods[j]]])$p.value, digits = 3)," now in ",i,j,"\n")
      compmatrix[i,j] = signif(pROC::roc.test(rc[[methods[i]]],rc[[methods[j]]])$p.value, digits = 3)
      cat("AUROC for ",fam," ", methods[i]," VS ",methods[j],": ",round(pROC::auc(rc[[methods[i]]])[1],4)," ",round(pROC::auc(rc[[methods[j]]])[1],4),"diff p-val:",signif(pROC::roc.test(rc[[methods[i]]],rc[[methods[j]]])$p.value, digits = 3),"\n")
    }
  }
  compmat = rbind(compmat,compmatrix)
  
  cat("***********\n")
  
  
  dft=data.frame(compmatrix)
  colnames(dft)=methods
  rownames(dft)=methods
  xtables[[toString(k)]]=xtable(dft,digits=cbind(rep(1,nrow(digitsfmt)),digitsfmt),caption=paste0("linear task ",k))
  #print(xtables$`1`, table.placement="H")
}
####################################################################################################

plotExperimentResults(pandoModel = mshared,perTaskModels = perTaskModels,postfix = "Linear",signalVars = 1:5)











# dd = matrix(coef(mgplasso))
# dd=matrix(dd)
# dd=data.frame(matrix(dd[-1,]))
# dd[,"varname"]=paste0(paste0("T",rep(1:5,each=26),"_"),rep(paste0("X",1:26),5))
# colnames(dd)=c("value","varname")
# dd <- dd[order(-abs(dd[,"value"])),]
# 
# pgp = ggplot(head(dd,n=40), aes(varname, weight = value)) + geom_bar()
# pgp = pgp + labs(title = paste("var importance gplasso"))
# pgp = pgp+ylab("linear coef")
# 
# pgp=pgp+thm
# pgp + coord_flip()
# pgp
# 
# 
# library(rpart.plot)
# bdata=binaryData[,-ncol(binaryData)]
# v=.1 
# fit=rpart(Label~.,data=bdata,controls=controls)
# yp=predict(fit)
# df$yr=df$y - v*yp
# YP=v*yp
# for(t in 1:100){
#   fit=rpart(yr~x,data=df)
#   yp=predict(fit,newdata=df)
#   df$yr=df$yr - v*yp
#   YP=cbind(YP,v*yp)
# }









