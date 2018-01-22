require(pROC)
require(rpart)
require(caret)
require(glmnet)
require(MASS)
require(partykit)
library(plyr)
library(ggplot2)
library(reshape2)
library(purge)

##tune a pando function with cross validation over the traindata:
## tuning rpart maxdepth,cp and num of iterations
## first tune tree params: maxdepth and cp on a fixed, probably samll number of iterations, then tune number of iterations
## for now we tune number of iterations on a validation set
TunePando = function(pandofunc,traindata,valdata,target="binary",maxiter=1000,cv=5){
  traindata = traindata[sample(nrow(traindata)),] ## shuffle 
  rate=0.1 # start with high leanring rate to tune the other parameters
  folds = createFolds(1:nrow(traindata),k=cv)
  iter=300
  grid = expand.grid(maxdepth=c(3,4,5,7,10),cp=c(0.01, 0.005))
  grid = cbind(grid,rep(NA,nrow(grid)))
  colnames(grid)[ncol(grid)]="bestCvScore"
  grid = cbind(grid,rep(NA,nrow(grid)))
  colnames(grid)[ncol(grid)]="bestCvIt"
  for(i in 1:nrow(grid)){
    cvpredictions = matrix(NA,nrow(traindata),ncol=iter-1) ### CV predictions per iterations 
    foldnum=0
    for(fold in folds){
      foldnum = foldnum+1
      train = traindata[fold,] ## train on this
      valdata = traindata[-fold,] ## predict on this
      maxdepth=grid[i,"maxdepth"]
      cp=grid[i,"cp"]

      controls = rpart.control(maxdepth = maxdepth,cp=cp)
      cat("fitting fold no ",foldnum ,"with params maxdepth=",maxdepth," cp=",cp,"\n")
      mshared=pandofunc(train,iter=iter,v=rate,groups=train[,"Family"],controls=controls,ridge.lambda=ridge.lambda,target="binary",valdata=valdata,earlystopping = -1)      
      #cvpredictions[-fold] = mshared$log$vpred[,iter-1] ## the predictions on the validation set after all the iteartions
      cvpredictions[-fold,1:(iter-1)] = mshared$log$vpred ## all predictions for this fold
    }
    aucs=apply(cvpredictions,2,function(x){as.numeric(pROC::auc(pROC::roc(as.factor(traindata[,"Label"]),as.numeric(x))))})
    bestCvIt = which(aucs==max(aucs))
    grid[i,"bestCvScore"]=max(aucs)
    grid[i,"bestCvIt"]=bestCvIt
  }
  bestGridIdx = which(grid[,"bestCvScore"]==max(grid[,"bestCvScore"]))
  bestParams = grid[bestGridIdx,]
  cat("found best parameters to be: maxdepth=",bestParams[,"maxdepth"], " cp=",bestParams[,"cp"],"\n")
  controls = rpart.control(maxdepth = bestParams[,"maxdepth"],cp=bestParams[,"cp"])
  ## now we can use the validation set to determine the number of iterations
  rate=0.01
  iter=1000
  cat("fitting pando with best parmaeter\n")
  mshared=pandofunc(train,iter=iter,v=rate,groups=train[,"Family"],controls=controls,ridge.lambda=ridge.lambda,target="binary",valdata=valdata,fitTreeCoef = FALSE,earlystopping = 100)      
  return(mshared)
}
#### plot feature importances and save locally:
plotExperimentResults = function(pandoModel,perTaskModels,outdir="", postfix="",signalVars=c()){
  if(outdir==""){
    outdir=getwd()
  }
  dff = FeatureImportance.BoostingModel(pandoModel$fam1) ### for pando, we can use only one family as they share the same tree structures 
  PlotImp(dff,signalVars = signalVars,flip = TRUE)
  filename=paste0(outdir,"/PandoFeatureImportance",postfix,".png")
  dev.copy(png,filename=filename)
  dev.off()
  
  
  dff = PerTaskImportances(perTaskModels) ### for ptb
  PlotImp(dff,signalVars = signalVars,flip = TRUE)
  filename=paste0(outdir,"/PTBFeatureImportance",postfix,".png")
  dev.copy(png,filename=filename)
  dev.off()
}



#### mock for group lasso
aa=matrix(1:18,ncol=3)
aa=data.frame(aa)
aa=cbind(aa,rep(1,nrow(aa)))
aa=cbind(aa,c("fam1","fam1","fam2","fam2","fam3","fam3"))
colnames(aa)=c("f1","f2","f3","Label","Family")

##########################################
scorefunc = function(label,preds,scoreType){
  if(scoreType=="rmse"){
    return(sqrt(mean((label-preds)**2)))
  }
  if(scoreType=="auc"){
    roc(as.factor(label),as.numeric(preds))$auc[1]    
  }
}


CreateGroupLassoDesignMatrix = function(X,interceptGrouped=FALSE,isIntercept=TRUE){
  
  families = unique(X[,"Family"])
  ntasks=length(families)
  colsPerTask = (ncol(X)-2) # removing label and family columns
  colsPerTask = colsPerTask+1 # adding intercept column per each task (two different rows of code just for readibility)
  Xgl = matrix(NA,nrow=nrow(X),ncol=(colsPerTask*ntasks)+1) # the +1 is for the label
  #cat(nrow(Xgl),"XGL",ncol(Xgl),"\n")
  i=0
  
  rowend=0
  #cat("ncol of xgl is:",ncol(Xgl),"\n")
  if(interceptGrouped){
    groups=rep(1:colsPerTask,ntasks) # which group each variable belongs to. in our case, we group the same variable across tasks  
  }else{
    groups=rep(c(1:(colsPerTask-1),0),ntasks) # which group each variable belongs to. in our case, we group the same variable across tasks  
  }
  
  Xgl=c()
  ygl=c()
  for(fam in families){
    cat("update gplasso matrix with fam is :" ,fam,"out of ",length(families)," families\n")
    taskX=as.matrix(X[X[,"Family"]==fam,-which(colnames(X) %in% c("Family","Label"))])
    taskX=cbind(taskX,matrix(1,nrow=nrow(taskX),ncol=1)) # add intercept per task
    tasky = as.matrix(X[X[,"Family"]==fam,"Label"],ncol=1)
    if(is.null(Xgl)){
      Xgl=taskX
      ygl=tasky
      next
    }
    taskXpad = matrix(0,ncol=ncol(Xgl),nrow=nrow(taskX))

    Xgl=cbind(Xgl,matrix(0,ncol=ncol(taskX),nrow=nrow(Xgl)))
    taskX = cbind(taskXpad,taskX)

    Xgl = rbind(Xgl,taskX)
    ygl = rbind(ygl,tasky) 

  }
  ret=list()
  cat(nrow(ygl),"\n")
  cat(nrow(Xgl),"\n")
  ret[["X"]]=cbind(Xgl,ygl)
  ret[["groups"]]=groups
  return(ret)
}


# binaryVariablesLambda 
CreateVibratingGroupLassoDesignMatrix = function(X,interceptGrouped=FALSE,isIntercept=TRUE,binaryVariablesLambda=1){
  
  families = unique(X[,"Family"])
  ntasks=length(families)
  colsPerTask = (ncol(X)-2) # removing label and family columns
  colsPerTask = colsPerTask+1 # adding intercept column per each task (two different rows of code just for readibility)
  Xgl = matrix(NA,nrow=nrow(X),ncol=(colsPerTask*ntasks)+1) # the +1 is for the label
  #cat(nrow(Xgl),"XGL",ncol(Xgl),"\n")
  i=0
  
  rowend=0
  #cat("ncol of xgl is:",ncol(Xgl),"\n")
  if(interceptGrouped){
    groups=rep(1:colsPerTask,ntasks) # which group each variable belongs to. in our case, we group the same variable across tasks  
  }else{
    groups=rep(c(1:(colsPerTask-1),0),ntasks) # which group each variable belongs to. in our case, we group the same variable across tasks  
  }
  
  Xgl=c()
  ygl=c()
  for(fam in families){
    cat("update gplasso matrix with fam is :" ,fam,"out of ",length(families)," families\n")
    taskX=as.matrix(X[X[,"Family"]==fam,-which(colnames(X) %in% c("Family","Label"))])
    taskX=cbind(taskX,matrix(1,nrow=nrow(taskX),ncol=1)) # add intercept per task
    tasky = as.matrix(X[X[,"Family"]==fam,"Label"],ncol=1)
    if(is.null(Xgl)){
      Xgl=taskX
      ygl=tasky
      next
    }
    taskXpad = matrix(0,ncol=ncol(Xgl),nrow=nrow(taskX))
    
    Xgl=cbind(Xgl,matrix(0,ncol=ncol(taskX),nrow=nrow(Xgl)))
    taskX = cbind(taskXpad,taskX)
    
    Xgl = rbind(Xgl,taskX)
    ygl = rbind(ygl,tasky) 
    
  }
  
  ## add a column per variable, across all tasks, to allow "binary lasso" to be on the regularization path
  binaryX = c()
  for(fam in families){
    binaryX = rbind(binaryX,as.matrix(X[X[,"Family"]==fam,-which(colnames(X) %in% c("Family","Label"))]))
  }
  ## add intercept
  binaryX = cbind(binaryX,matrix(1,nrow=nrow(binaryX),ncol=1))
  
  ## express binary regularization lambda
  binaryX = binaryVariablesLambda * binaryX
  Xgl = (1-binaryVariablesLambda)*Xgl ## balance between binary solution and group lasso solution
  Xgl = cbind(Xgl,binaryX)
  groups = c(groups,unique(groups)) ## binary variables are not penalized by group lasso
  ### in order to express 
  
  ret=list()
  cat(nrow(ygl),"\n")
  cat(nrow(Xgl),"\n")
  
  ret[["X"]]=cbind(Xgl,ygl)
  ret[["groups"]]=groups
  return(ret)
}

## preds = predictions, y = true value
negBinLogLikeLoss = function(preds=preds,y=y){
  ret = log(1+exp(-2*y*preds))
  return(ret)
}

## preds = predictions, y = true value
squareLoss = function(preds,y){
  ret = (preds-y)**2
  return(ret)
}

### return the negative gradient with respect to the loss function, 
### will be simply residuals for least squares
negative_gradient = function(y,preds,groups=NULL,target="binary",unbalanced=FALSE){
  #####
  ##
  if(target=="binary"){
    preds0 = 1-preds
    preds0[preds0<0.00001]=0.00001 ## not allow very small divisions
    preds[preds<0.00001]=0.00001 ## not allow very small divisions
    
    # if(unbalanced){
    #   
    #   Iplus = as.numeric(y==1)
    #   nplus = sum(Iplus)
    #   Iminus = as.numeric(y==-1)
    #   nminus = sum(Iminus)
    #   ret = preds*((Iplus/nplus) + (Iminus/nminus))
    # }else{
    ff = 0.5*log(preds/preds0)
    ret = (2*y)/(1+exp(2*y*ff)) ## greedy function approximation, a gradient boosting machine, page 9
  # }
    #####
    
  }else if(target=="regression"){
    ret = y-preds
  }
  return(ret)
}

negative_gradient2 = function(y,preds,groups=NULL){
  ret = (2*y)/(1+exp(2*y*preds)) ## A gradient boosting machine, page 9
  return(ret)
}





TreeWithCoef= function(treeFit,fittedCoef,intercept,treeType="rpart") {
  ### if we make the returned object from here
  ### compatible with "predict", we can 
  ### use the predict.boost
  treeFit = purge(treeFit)
  model = structure(list(treeFit=treeFit,fittedCoef=fittedCoef,intercept=intercept,treeType=treeType),class="treeWithCoef")
  return(model)
  
}

predict.treeWithCoef = function(modelObject,newdata){
  fit=modelObject$treeFit
  treeType=modelObject$treeType
  if(treeType=="rpart"){
    preds = predict(fit,newdata)    
  }else{
    preds = predict(fit,data.frame(x=newdata))  
  }
  
  
  #### TODO: change classification instance of the code to work with an underlying regression tree
  # if(fit$method=="class"){  # i think that ANYWAY we never use classification trees, behind the scenes we only do regression trees?
  #   preds=data.frame(preds[,1]) #  this takes the probability to be in class 1  
  # }
  ret=(preds*modelObject$fittedCoef)+(modelObject$intercept)

  return(ret)
}
  
### given a tree fit, return a function which given data, predicts the
### tree prediction and multiplies by the appropriate coefs according
### to which leaf the prediction falls in
TreeWithLeafCoefs= function(treeFit,leafToCoef) {
  ### if we make the returned object from here
  ### compatible with "predict", we can 
  ### use the predict.boost
  treeFit=purge(treeFit)
  model = structure(list(treeFit=treeFit,leafToCoef=leafToCoef),class="treeWithLeafCoefsModel")
  return(model)
  
}


predict.treeWithLeafCoefsModel = function(modelObject,newdata){
  
  X=newdata
  
  fit = modelObject$treeFit
  leafToCoef = modelObject$leafToCoef
  ## using the partykit package, we can get nodes for exsiting rpart object,
  ## i chekced and it corresponds exactly with rpart
  predNodes=rpart:::pred.rpart(fit, rpart:::rpart.matrix(X))
  preds = predict(fit,newdata=X) ##prediction for all X
  if(fit$method=="class"){
    preds=data.frame(preds[,1]) #  this takes the probability to be in class 1  
  }
  
  
  nodeValues = unique(predNodes)
  predCoefs = rep(0,length(preds))
  for(node in nodeValues){
    ##prepare for each prediction, by which coefficient it should be multiplied
    ##if it belongs to $node, multiply it by the matching coefficient
    if(is.na(leafToCoef[[node]])){
      #leafToCoef[[node]]=1.0
      stop(TRUE)
    }
    predCoefs[predNodes==node]=leafToCoef[[node]]
    
  }
  #cat(predCoefs,"\n")
  return(predCoefs)
  
}


BoostingModel= function(model,rate) {
  ## generic additive model. model should be a list of models, their prediction on data will be the added 
  ## value
  model = structure(list(modelList=model,rate=rate),class="BoostingModel")
  return(model)
  
}

predict.BoostingModel = function(m,X,calibrate=TRUE,bestIt=NULL,rate=NULL){
  if(is.null(rate)){
    rate=m$rate  
  }

  ## first, for each of the fitted sub models, create a prediction at X
  pred=rep(m$modelList[[1]],nrow(X)) ## fill with the initial guess
  if(is.null(bestIt)){
    bestIt=length(m$modelList)
  }
  if(bestIt>1){
    for(i in 2:bestIt){
      if(i > length( m$modelList)){
        cat("model is length ", length(m$modelList)," i is ", i,"\n")
      }
      mm = m$modelList[[i]]  # extract i-th tree
      newpred=predict(modelObject=mm,newdata=X)
      pred = pred+(rate*newpred)
      #pred = pred+newpred
      pred = as.matrix(pred,ncol=1)
      if(nrow(pred) != nrow(X)){
        cat("predict in submodel yielded different number of rows\n")
      }
    }
  }
  if(calibrate){
    pred = 1/(1+exp(-2*pred)) ## convert to logistic score  
  }
  return(pred)
  
}









LogLoss<-function(actual, predicted)
{
  actual[actual==-1]=0 ## fix if we got {-1,1} instead of {0,1}, otherwise will do nothing
  result<- -1/length(actual)*(sum((actual*log(predicted)+(1-actual)*log(1-predicted))))
  return(result)
}




BoostingModelFeatureImportance = function(model){
  n = length(model)
  importances=list()
  sum=0
  for(i in 2:n){
    imp = model$modelList[[i]]$treeFit$variable.importance
    
    for(v in names(imp)){
      if(is.null(importances$v)){
        importances[[v]]=0
      }
      importances[[v]]= importances[[v]]+as.numeric(imp[v])
    }
  }
  return(importances)
}




featureImp.BoostingModel = function(m){
  tt1=BoostingModelFeatureImportance(m)
  return(tt1)
}

FeatureImportance.BoostingModel = function(m,title=""){
  imp1=featureImp.BoostingModel(m) # dispatch feature importance
  dff <- melt(ldply(imp1, data.frame))
  dff=dff[,-2]
  colnames(dff)=c("varname","value")
  dff=dff[order(-dff[,"value"]),]
  dffnorm=dff
  dffnorm[,"value"]=dffnorm[,"value"]/sum(dffnorm[,"value"])
  colnames(dffnorm)=c("varname","value")
  #dffnorm <- base::transform(dffnorm, varname = reorder(varname, -value))
  dffnorm=dffnorm[order(-dffnorm[,"value"]),]
  return(dffnorm)
}

 


PerTaskImportances=function(perTaskModels){
  allimp=NULL
  for(fam in names(perTaskModels)[grepl("fam",names(perTaskModels))]){
    dff=FeatureImportance.BoostingModel(perTaskModels[[fam]][[fam]])
    if(is.null(allimp)){
      allimp = dff
      next
    }
    modelFeatures=levels(dff[,"varname"])
    for(var in modelFeatures){
      if(var %in% allimp[,"varname"]){
        allimp[allimp[,"varname"]==var,"value"] = as.numeric(allimp[allimp[,"varname"]==var,"value"])+dff[dff[,"varname"]==var,"value"]
      }else{
        newvar=matrix(c(var,dff[dff[,"varname"]==var,"value"]),nrow=1,ncol=2)
        colnames(newvar)=c("varname","value")
        allimp=rbind(allimp,newvar)
      }
    }
  }  
  dffnorm=allimp
  dffnorm[,"value"]=as.numeric(dffnorm[,"value"])/sum(as.numeric(dffnorm[,"value"]))
  dffnorm <- dffnorm[order(-dffnorm[,"value"]),]
  return(dffnorm)
}




## df = feature importance dataframem generated by FeatureImportance.BoostingModel
## for pando we used FeatureImportance.BoostingModel(mshared$fam1) to generate df (as it doesn't matter which family (they all use the same trees))
## for per task boosting models we use PerTaskImportances(perTaskModels) to get a df which reflects joint feature importance across tasks
PlotImp = function(df,signalVars=c(), title="",  flip=FALSE, nfirstvar=30){
  df <- base::transform(df, varname = reorder(varname, if(flip) value else -value))
  if(length(signalVars)==0){
    signalVarNames = df[,"varname"]
  }else{
    signalVarNames = c(paste0("X",signalVars))  
  }
  df[,"Legend"]="noise"
  df[df[,"varname"] %in% signalVarNames,"Legend"]="signal"
  group.colors <- c(noise = "black", signal = "grey") 
  p3 = ggplot(head(df,n=nfirstvar), aes(varname, weight = value,fill=Legend)) + geom_bar()
  p3 = p3+scale_fill_manual(values=group.colors)
  p3 = p3 + labs(title = title)
  p3 = p3+ylab("split gain")
  if(flip){
    p3=p3+coord_flip()
  }
  p3  
}