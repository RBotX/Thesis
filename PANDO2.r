source("helper.R")
TrainMultiTaskClassificationGradBoost2 = function(df,iter=3,v=1,groups,controls,ridge.lambda,target="binary",df.val=NULL,fitCoef="ls"){
  
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
        cat("train RMSE is:",sqrt(mean((data$Label-yp)**2)),"\n")  
        #cat("train AUC is:")#,pROC::auc(yp,data$label),"\n")  
      }
    }
    
    #cat(head(yp,n=50),"-----------\n")
    pr = negative_gradient(y=data$Label,preds=yp,target=target) ## as if y-yp but multiply each adition by v so it's y-v*yp
    if(any(is.na(pr))){
      cat("pr is na2\n")
    }
    
    ## create a tree for all families together, 1 vs 0
    #fit=ctree(y~.,data=data.frame(x=data[,-which(colnames(data) %in% c("Label","Family"))],y=pr),control=controls,cores=3)
    fit=rpart(pr~.,data=data[,-which(colnames(data) %in% c("Label","Family"))],control=controls,method="anova")
    ridgeRegX = NULL
    ridgeRegy = NULL
    for(fam in families){
      ###  fit a coefficient per entire tree
      famx = data[(data[,"Family"]==fam),-which(colnames(data) %in% c("Label","Family"))]
      famX=predict(fit,famx)
      #famX=predict(fit,data.frame(x=famx))
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
      fittedIntercept = 0
      
      if(fitCoef == "norm2"){
        fittedCoef = sqrt(sum((lmdf[,"x"]-lmdf[,"y"])**2)) ## like in obozinski  
      }
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
      if(any(is.na(pp))){
        cat("fam pred na\n")
      }
      
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