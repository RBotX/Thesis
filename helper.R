require(pROC)
require(rpart)
require(caret)
require(glmnet)
require(MASS)
require(partykit)

##########################################
CreateGroupLassoDesignMatrix = function(X){
  cat("hey")
  families = unique(X[,"Family"])
  ntasks=length(families)
  colsPerTask = (ncol(X)-2) # removing label and family columns
  colsPerTask = colsPerTask+1 # adding intercept column per each task (two different rows of code just for readibility)
  Xgl = matrix(-123456,nrow=nrow(X),ncol=(colsPerTask*ntasks)+1) # the +1 is for the label
  #cat(nrow(Xgl),"XGL",ncol(Xgl),"\n")
  i=0
  
  rowend=0
  #cat("ncol of xgl is:",ncol(Xgl),"\n")
  groups=rep(1:colsPerTask,ntasks) # which group each variable belongs to. in our case, we group the same variable across tasks
  for(fam in families){
    #cat("fam is :" ,fam,"\n")
    nr = nrow(X[X[,"Family"]==fam,]) ## how many instances for this family/task
    nc = colsPerTask ## we know in our formulation that all tasks share the same number of features
    rowstart=(rowend+1)#(i*nr)+1
    rowend = rowstart + nr -1
    colstart = (i*nc)+1
    colend = (i+1)*nc
    #cat(rowstart,"->",rowend,"\n")
    #cat(colstart,"->",colend,"\n")
    uu=as.matrix(X[X[,"Family"]==fam,-which(colnames(X) %in% c("Family","Label"))])
    Xgl[rowstart:rowend,colstart:(colend-1)]<-uu
    Xgl[rowstart:rowend,colend]=matrix(1,nrow=length(rowstart:rowend),ncol=1) ## add intercept per task
    ## add label
    Xgl[rowstart:rowend,ncol(Xgl)] = as.matrix(X[X[,"Family"]==fam,"Label"])
    i=i+1
    
  }
  ret=list()
  ret[["X"]]=Xgl
  ret[["groups"]]=groups
  return(ret)
}


negBinLogLikeLoss = function(preds,y){
  ret = log(1+exp(-2*y*preds))
  return(ret)
}
### return the negative gradient with respect to the loss function, 
### will be simply residuals for least squares
negative_gradient = function(y,preds,groups=NULL,target="binary"){
  #####
  ##
  if(target=="binary"){
    preds0 = 1-preds
    preds0[preds0<0.001]=0.001 ## not allow very small divisions
    preds[preds<0.001]=0.001 ## not allow very small divisions
    ff = 0.5*log(preds/preds0)
    ret = (2*y)/(1+exp(2*y*ff)) ## A gradient boosting machine, page 9
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





TreeWithCoef= function(treeFit,fittedCoef,intercept) {
  ### if we make the returned object from here
  ### compatible with "predict", we can 
  ### use the predict.boost
  
  model = structure(list(treeFit=treeFit,fittedCoef=fittedCoef,intercept=intercept),class="treeWithCoef")
  return(model)
  
}

predict.treeWithCoef = function(modelObject,newdata){
  fit=modelObject$treeFit
  preds = predict(fit,data.frame(x=newdata))
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

predict.BoostingModel = function(m,X,calibrate=TRUE){
  
  rate=m$rate
  ## first, for each of the fitted sub models, create a prediction at X
  pred=rep(m$modelList[[1]],nrow(X)) ## fill with the initial guess
  for(i in 2:length(m$modelList)){
    mm = m$modelList[[i]]  # extract i-th model
    newpred=predict(modelObject=mm,newdata=X)
    pred = pred+(rate*newpred)
    #pred = pred+newpred
    pred = as.matrix(pred,ncol=1)
    if(nrow(pred) != nrow(X)){
      cat("predict in submodel yielded different number of rows\n")
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
    imp = model[[i]]$treeFit$variable.importance
    
    for(v in names(imp)){
      if(is.null(importances$v)){
        importances[[v]]=0
      }
      importances[[v]]= importances[[v]]+as.numeric(imp[v])
    }
  }
  return(importances)
}

plotFeatureImportace = function(imp){
  
}


