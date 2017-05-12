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



# MultiTaskBoostingModel = function(X,y,iter,rate,groups,controls,target,lambda.ridge=1){
#   model = structure(list(X=X,y=y,iter=iter,rate=rate,groups=groups,controls=controls,lambda.ridge=1),class="MultiTaskBoostingModel")
#   return(model)
# }



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








###*********************************************************************
TrainMultiTaskClassificationGradBoost2 = function(df,iter=3,v=1,groups,controls,ridge.lambda,target="binary",df.val=NULL){
  
  families = unique(groups)[unique(groups)!="clean"]
  data = df  
  finalModel=list()
  
  if(target=="binary"){
    preds = 0.5 * log( (1+mean(data$Label))/(1-mean(data$Label))  ) ### initial model  
  }else{
    preds = 0#mean(data$Label)
  }
  
  finalModel[[1]]=preds
  yp = rep(preds,nrow(data)) ## initial guess without learning rate?
  #numFamilies = length(unique(groups))-1 ## clean doesn't count as family
  numFamilies = length(unique(groups)) ## clean doesn't count as family
  finalModel[["rate"]]=v
  for(fam in families){
    finalModel[[toString(fam)]]=list()
    finalModel[[toString(fam)]][[1]] = preds
    
    
  }
  
  for(t in 2:iter){
    ## for each new tree, we have a new leaf->coef per family
    for(fam in families){
      leavesToCoefs = list()
      leavesToCoefs[[toString(fam)]]=list() 
    }
    
    ### pseudo responses
    if(t%%20 == 0){
      cat("iteration ",t,"\n")
      if(target=="regression"){
        cat("train RMSE is:",sqrt(mean((data$Label-yp)**2)),"\n")  
      }else{
        cat("train AUC is:")#,pROC::auc(yp,data$label),"\n")  
      }
    }
    
    #cat(head(yp,n=50),"-----------\n")
    pr = negative_gradient(y=data$Label,preds=yp,target=target) ## as if y-yp but multiply each adition by v so it's y-v*yp
    if(any(is.na(pr))){
      cat("pr is na2\n")
    }
    
    ## create a tree for all families together, 1 vs 0
    #fit=rpart(pr~.,data=data[,-which(colnames(data) %in% c("Label","Family"))],control=controls,method=if(target=="binary") "class" else "anova")
    #fit=ctree(pr~.,data=data[,-which(colnames(data) %in% c("Label","Family"))],control=ctree_control())
    fit=ctree(y~.,data=data.frame(x=data[,-which(colnames(data) %in% c("Label","Family"))],y=pr),control=ctree_control(maxdepth = 3),cores=3)
    ridgeRegX = NULL
    ridgeRegy = NULL
    for(fam in families){
      ###  fit a coefficient per entire tree
      famx = data[(data[,"Family"]==fam),-which(colnames(data) %in% c("Label","Family"))]
      #famX=predict(fit,famx)
      famX=predict(fit,data.frame(x=famx))
      famy = pr[(data[,"Family"]==fam)]
      lmdf = matrix(ncol=2,nrow=length(famX))
      lmdf[,1]=as.matrix(famX,ncol=1)
      lmdf[,2]=as.matrix(famy,ncol=1)
      colnames(lmdf)=c("x","y")
      
      lmdf = data.frame(lmdf)
      if(!is.null(ridgeRegX)){
        newx = lmdf[,"x"]
        newx = rbind(matrix(0,nrow=nrow(ridgeRegX),ncol=1),matrix(newx,ncol=1))
        newy = matrix(lmdf[,"y"],ncol=1)
        ridgeRegX = rbind(ridgeRegX,matrix(0,ncol = ncol(ridgeRegX),nrow=nrow(lmdf)))
        ridgeRegX = cbind(ridgeRegX,newx)
        ridgeRegy = rbind(ridgeRegy,newy)
      }else{
        ridgeRegX = matrix(lmdf[,"x"],ncol=1)
        ridgeRegy = matrix(lmdf[,"y"],ncol=1)
      }
      
      
      mm = lm(y~x -1,data=lmdf)

      fittedCoef = as.numeric(coef(mm)[1])
      if(is.na(fittedCoef)){
        fittedCoef=1
      }
      fittedIntercept = 0#as.numeric(coef(mm)[1]) ### fitting without intercept
      #cat("fitted coef is: ",fittedCoef,"\n")
      finalModel[[toString(fam)]][[t]] = TreeWithCoef(fit,fittedCoef,fittedIntercept)
      
    }
    ridgeReg = cbind(ridgeRegX,ridgeRegy)
    ridgeReg=data.frame(ridgeReg)
    colnames(ridgeReg)=c(families,"y")

    # #mm = lm.ridge(y~-1.,data = ridgeReg,lambda=ridge.lambda,tol=0.001)
    # mm=glmnet(as.matrix(ridgeReg[,-which(colnames(ridgeReg) == "y")]),as.matrix(ridgeReg[,"y"]), alpha = 0, lambda = ridge.lambda,intercept=FALSE)
    # for(i  in 1:length(families)){
    #   fittedCoef = coef(mm)[i]
    #   fittedIntercept=0 ### fitting without intercept
    #   finalModel[[toString(families[i])]][[t]] = TreeWithCoef(fit,fittedCoef,fittedIntercept)      
    # }
    
    ## generate new pseduo-responses:
    famPreds=matrix(ncol=1,nrow=length(yp))
    for(fam in families){
      pp = predict(finalModel[[toString(fam)]][[t]],data[data[,"Family"]==fam,-which(colnames(data) %in% c("Label","Family"))])
      famPreds[data[,"Family"]==fam,1]=as.matrix(pp,ncol=1)
    }
    yp = yp + v*famPreds
    #val.fam.preds = val.preds + v*
  }
  
  ret=list()
  for(fam in families){
    ret[[toString(fam)]] = BoostingModel(finalModel[[toString(fam)]],rate=rate)
  }
  ret[["rate"]]=v
  return(ret)  
}

###*************************************************************************

##***************************************************************************
TrainMultiTaskClassificationGradBoost = function(df,iter=3,v=1,groups,controls,ridge.lambda,target="binary"){
  families = unique(groups)[unique(groups)!="clean"]
  data = df  
  finalModel=list()
  
  if(target=="binary"){
    preds = 0.5 * log( (1+mean(data$Label))/(1-mean(data$Label))  ) ### initial model  
  }else{
    preds = 0#mean(data$Label)
  }
  
  finalModel[[1]]=preds
  yp = rep(preds,nrow(data)) ## initial guess without learning rate?
  #numFamilies = length(unique(groups))-1 ## clean doesn't count as family
  numFamilies = length(unique(groups)) ## clean doesn't count as family
  finalModel[["rate"]]=v
  for(fam in families){
    finalModel[[toString(fam)]]=list()
    finalModel[[toString(fam)]][[1]] = preds
    
    
  }
  
  for(t in 2:iter){
    ## for each new tree, we have a new leaf->coef per family
    for(fam in families){
      leavesToCoefs = list()
      leavesToCoefs[[toString(fam)]]=list() 
    }
    
    ### pseudo responses
    if(t%%20 == 0){
      cat("iteration ",t,"\n")
      if(target=="regression"){
        cat("train RMSE is:",sqrt(mean((data$Label-yp)**2)),"\n")  
      }else{
        cat("train AUC is:")#,pROC::auc(yp,data$label),"\n")  
      }
    }
    
    #cat(head(yp,n=50),"-----------\n")
    pr = negative_gradient(y=data$Label,preds=yp,target=target) ## as if y-yp but multiply each adition by v so it's y-v*yp
    
    ## create a tree for all families together, 1 vs 0
    fit=rpart(pr~.,data=data[,-which(colnames(data) %in% c("Label","Family"))],control=controls,method="anova")
    
    
    ### fit model with per leaf score
    leaves = unique(fit$where)
    for(l in leaves){
      
      samplesInLeaf = (fit$where==l) ## which are in the l-th leaf
      if(length(which(samplesInLeaf))==0){
        next
      }
      
      
      ### take leaf predictions in that leaf, treat those as a variable in a ridge prediction
      ridgeRegX = matrix(nrow=0,ncol=numFamilies)
      ridgeRegy = matrix(nrow=0,ncol=1)
      PadColsToLeft = 0
      PadColsToRight = numFamilies-1
      for(fam in families){
        ## check if this family has results in this leaf, if not, skip
        #cat(fam," ",length(which((samplesInLeaf)&(data[,"Family"]==fam))) == 0,"\n")
        
        if(length(which((samplesInLeaf)&(data[,"Family"]==fam))) == 0){
          
          next
        }
        
        #X = predict(fit,data[(samplesInLeaf)&(data[,"Family"]==fam),-which(colnames(data) %in% c("Label","Family"))]) ### predictions for this family and clean in this leaf
        #X = as.matrix(X,nrow=length(x),ncol=1)
        
        
        #y = negative_gradient(y=(data$Label)[(samplesInLeaf)&(data[,"Family"]==fam)],preds=X+yp[(samplesInLeaf)&(data[,"Family"]==fam)]) ### pseudo responses for this family and clean in this leaf
        y = pr[(samplesInLeaf)&(data[,"Family"]==fam)]
        y = matrix(y,nrow=length(y),ncol=1)


        X = matrix(1,nrow=nrow(y),ncol=1) ### 1 for each observation in this leaf for this family
        
        
        
        if(PadColsToLeft > 0){
          X = cbind(matrix(0,nrow=nrow(X),ncol=PadColsToLeft),X)
        }
        if(PadColsToRight > 0){
          X = cbind(X,matrix(0,nrow=nrow(X),ncol=PadColsToRight))
        }
        PadColsToLeft = PadColsToLeft+1
        PadColsToRight = PadColsToRight-1
        ridgeRegX = rbind(ridgeRegX,X)
        ridgeRegy = rbind(ridgeRegy,y)
      }
      y=ridgeRegy
      colnames(y)="y"
      colnames(ridgeRegX)=families
      nonZeroFams= c()
      cnames=colnames(ridgeRegX)
      for(cname in cnames){
        #cat(paste("dim ridgeRegX",dim(ridgeRegX),"\n"))
        if(is.null(dim(ridgeRegX))){
          ridgeRegX=t(as.matrix(ridgeRegX))
        }
        if(all(ridgeRegX[,cname]==0)){
          
          ridgeRegX = ridgeRegX[,-which(colnames(ridgeRegX)==cname)]
        }else{
          nonZeroFams = c(nonZeroFams,cname)
        }
        
      }
      ## when we create a matrix of two columns, and remove one column, we get that ncol=null.
      ## however, when we get a 
      if(is.null(ncol(ridgeRegX)) | numFamilies==1){ ## we have a single family in that leaf
        
        # fam = nonZeroFams
        # leafValue = mean(y)
        # leavesToCoefs[[fam]][[l]]=leafValue
        # next ### finish for this leaf
        for(fam in families){
          leavesToCoefs[[toString(fam)]][[l]]=mean(y)
        }
        next
      }
      
      PerLeafData = data.frame(ridgeRegX)
      PerLeafData = cbind(PerLeafData,y)
      #cat("before ridge target is ",head(y),"***********\n")
      m = lm.ridge(y~.-1,data = PerLeafData,lambda=ridge.lambda) 
      leafCoefs=coef(m)
      #cat("ceofs of ridge:",coef(m),"\n")
      for(fam in families){
        if(fam %in% names(leafCoefs)){
          intercept = as.numeric(coef(m)[1])
          #leavesToCoefs[[toString(fam)]][[l]]=as.numeric(leafCoefs[toString(fam)])+intercept
          leavesToCoefs[[toString(fam)]][[l]]=as.numeric(leafCoefs[toString(fam)])
        }else{
          leavesToCoefs[[toString(fam)]][[l]]=mean(y)
        }
      }
    }
    ### for this stage, we have a tree per family, and a coefficient per leaf per tree.
    ### in the final model, we can now have the TreeWithLeafWeights object and use this
    ### when we predic in the predict.boost stage...
    for(fam in families){
      
      finalModel[[toString(fam)]][[t]] = TreeWithLeafCoefs(fit,leavesToCoefs[[toString(fam)]])      
    }
    
    ## generate new pseduo-responses:
    
    #lastFamPredsLength=0
    # for(fam in families){
    #   famPreds = predict(finalModel[[toString(fam)]][[t]],data[data[,"Family"]==fam,-which(colnames(data) %in% c("Label","Family"))])       
    #   #cat(head(famPreds),"\n")
    #   famPreds = as.matrix(famPreds,nrow=length(famPreds),ncol=1)
    #   lastFamPredsLength=lastFamPredsLength+length(famPreds)
    #   if(lastFamPredsLength>length(famPreds)){
    #     zeroesToPad = lastFamPredsLength-length(famPreds)
    #     famPreds = rbind(matrix(0,nrow=zeroesToPad,ncol=1),famPreds)
    #     
    #   }
    #   if(length(famPreds)<length(yp)){
    #     zeroesToPad = length(yp)-length(famPreds)
    #     famPreds=rbind(famPreds,matrix(0,nrow=zeroesToPad,ncol=1))
    #   }
    #   
    #   yp = yp + v*famPreds 
    # }
    famPreds=matrix(ncol=1,nrow=length(yp))
    for(fam in families){
      pp = predict(finalModel[[toString(fam)]][[t]],data[data[,"Family"]==fam,-which(colnames(data) %in% c("Label","Family"))])
      famPreds[data[,"Family"]==fam,1]=as.matrix(pp,ncol=1)
    }
    yp = yp + v*famPreds
    
  }
  #return(finalModel)  
  ret=list()
  for(fam in families){
    ret[[toString(fam)]] = BoostingModel(finalModel[[toString(fam)]],rate=rate)
  }
  ret[["rate"]]=v
  return(ret)  
  
}
##***************************************************************************



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


# 
# ################################################################################################
# ################################################################################################
# ################################################################################################
# ################################################################################################
# ################################################################################################
# ################################################################################################
# ################################################################################################
# ################################################################################################
# ################################################################################################
