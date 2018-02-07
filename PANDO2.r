source("helper.R")

### fit a coefficient per tree per task. all tasks share the same trees with different coefficients
TrainMultiTaskClassificationGradBoost2 = function(df,valdata=NULL,earlystopping=100,iter=3,v=1,groups,controls,target="binary",fitCoef="ridge",treeType="rpart",fitTreeCoef = NULL, fitLeafCoef = NULL,unbalanced=FALSE){
  
  scoreType = if(target == "binary") "auc" else "rmse"
  log=list()
  log[["tscore"]]=c()
  log[["vscore"]]=c()
  log[["vpred"]]=c()
  #families = unique(groups)[unique(groups)!="clean"]
  #families = paste0("fam",0:(length(families)-1))
  families=unique(groups)
  data = df  
  finalModel=list()
  isval=!is.null(valdata)
  if(target=="binary"){
    preds = 0.5 * log( (1+mean(data$Label))/(1-mean(data$Label))  ) ### initial model  
  }else{
    preds = mean(data$Label)
  }
  
  finalModel[[1]]=preds
  yp = rep(preds,nrow(data)) ## initial guess without learning rate?
  ypscore=yp
  if(isval){
    ypval = rep(preds,nrow(valdata)) ## initial guess without learning rate?
    ypvalscore= ypval
    bestVscore=scorefunc(label=valdata$Label,preds=ypvalscore,scoreType=scoreType)
  }
  
  numFamilies = length(unique(groups)) ## clean doesn't count as family
  finalModel[["rate"]]=v
  for(fam in families){
    finalModel[[toString(fam)]]=list()
    finalModel[[toString(fam)]][[1]] = preds
    
  }
  bestScoreRound=1
  t=1
  while(t<iter){
    t=t+1
    if((isval)&(t-bestScoreRound > earlystopping)&(earlystopping>0)){
      cat("EARLY STOPPING AT ",t," best iteration was ",bestScoreRound," with validation score ",bestVscore,"\n")
      break
    }
    
    tscore = scorefunc(label=data$Label,preds=yp,scoreType=scoreType)
    log[["tscore"]]=c(log[["tscore"]],tscore)
    if(!is.null(valdata)){
        # vscore=0
        # for(fam in families){
        #   idxs=(valdata[,"Family"]==fam)
        #   vscore =vscore+ scorefunc(label=valdata$Label[idxs],preds=ypvalscore[idxs],scoreType=scoreType)  
        # }
        # vscore = vscore/length(families) ## average scores
        vscore = scorefunc(label=valdata$Label,preds=ypvalscore,scoreType=scoreType)  
        if(((vscore > bestVscore)&(scoreType == "auc"))||((vscore < bestVscore)&(scoreType == "rmse"))){
          bestVscore = vscore
          bestScoreRound=t
        }
        
        log[["vscore"]]=c(log[["vscore"]],vscore)
        log[["vpred"]]=cbind(log[["vpred"]],ypvalscore)
    }
  
    
    ### pseudo responses
    if(t%%20 == 0){
      cat("iteration ",t,"\n")
      if(isval){
        cat("valscore: ",vscore,"\n----------------\n")
      }
      
    }
    
    #cat(head(yp,n=50),"-----------\n")
    pr = negative_gradient(y=data$Label,preds=ypscore,target=target,unbalanced=unbalanced) ## as if y-yp but multiply each adition by v so it's y-v*yp  
    
    
    if(any(is.na(pr))){
      cat("pr is na2\n")
    }
    
    ## create a tree for all families together, 1 vs 0
    if(treeType=="rpart"){
      fit=rpart(pr~.,data=data[,-which(colnames(data) %in% c("Label","Family"))],control=controls,method="anova")
      environment(fit$terms) <- NULL ## shrink size of rpart opbject
      #fit = purge(fit)
    }else{
      fit=ctree(y~.,data=data.frame(x=data[,-which(colnames(data) %in% c("Label","Family"))],y=pr),control=controls,cores=3)  
    }
    
    ridgeRegX = NULL
    ridgeRegy = NULL
    if(fitCoef=="ridge" | fitCoef=="obo"){
      for(fam in families){
        ###  fit a coefficient per entire tree per family
        
        
          famx = data[(data[,"Family"]==fam),-which(colnames(data) %in% c("Label","Family"))]
          if(treeType=="rpart"){
            famX=predict(fit,famx)  
          }else{
            famX=predict(fit,data.frame(x=famx))  
          }
          ## famX is the predictions per family
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
          
          ridgeReg = cbind(ridgeRegX,ridgeRegy)
        }
        ridgeReg=data.frame(ridgeReg)
        colnames(ridgeReg)=c(families,"y")
  
        if(fitCoef=='ridge')
        {
          useglmnet=FALSE
          lambdas = 2^seq(3, -10, by = -.1)
          if(useglmnet){
            ridgeReg = cbind(ridgeRegX,ridgeRegy)
            ridgeReg=data.frame(ridgeReg)
            colnames(ridgeReg)=c(families,"y")
            
            mm=cv.glmnet(as.matrix(ridgeReg[,-which(colnames(ridgeReg) == "y")]),as.matrix(ridgeReg[,"y"]), alpha = 0, nlambda = 100,intercept=FALSE,nfolds=10,standardize=T)          
            coefs =coef(mm,s="lambda.min")
            coefs = data.frame(as.matrix(coefs)[-1,])[,1]
            
          }else{
            m = lm.ridge(y~.-1,data = ridgeReg,lambda=lambdas) 
            whichIsBest <- which.min(m$GCV) 
            coefs = coef(m)[whichIsBest,]
          }
          names(coefs)=families
        }
      }
      obocoefs=NULL
      if(fitCoef == "obo"){        
        ## the overall gradient
        gradpertask=c()
        for(fam in families){
          taskidx=(data[,"Family"]==fam)
          taskgrad=-negative_gradient(ridgeReg[taskidx,"y"],ridgeReg[taskidx,fam],target)
          gradpertask=c(gradpertask,mean(taskgrad)) ## approximated task gradient (mean)
        }
        names(gradpertask)=families
        gradnorm = sqrt(sum(gradpertask**2)) ## norm of the gradient per task vector
        obocoefs = -gradpertask/gradnorm ## obozinski coefficient
      }
    
      for(fam in families){
        if(fitCoef == "obo"){
          fittedCoef=as.numeric(obocoefs[fam])
        }
        
        if(fitCoef == "nocoef"){
          fittedCoef=1 ## standard boosting
        }
        
        if(fitCoef == "ridge"){
          fittedCoef = as.numeric(coefs[fam])          
        }
        
        fittedIntercept = 0
        finalModel[[toString(fam)]][[t]] = TreeWithCoef(fit,fittedCoef,fittedIntercept,treeType=treeType)
        
      }
      
    
    
    ## generate new pseduo-responses:
    famPreds=matrix(ncol=1,nrow=length(yp))
    if(isval){
      valfamPreds=matrix(ncol=1,nrow=length(ypval))  
    }
    
    for(fam in families){
      pp = predict(finalModel[[toString(fam)]][[t]],data[data[,"Family"]==fam,-which(colnames(data) %in% c("Label","Family"))])
      if(isval){
        ppval = predict(finalModel[[toString(fam)]][[t]],valdata[valdata[,"Family"]==fam,-which(colnames(data) %in% c("Label","Family"))])  
      }
      
      if(any(is.na(pp))){
        cat("fam pred na\n")
      }
      
      famPreds[data[,"Family"]==fam,1]=as.matrix(pp,ncol=1)
      if(isval){
        valfamPreds[valdata[,"Family"]==fam,1]=as.matrix(ppval,ncol=1)  
      }
      
    }
    yp = yp + v*famPreds
    if(isval){
      ypval = ypval + v*valfamPreds
    }
    if(target=="binary"){ ## calibrate predictions if binary
      ypscore = 1/(1+exp(-2*yp)) ## convert to logistic score  
      if(isval){
        ypvalscore = 1/(1+exp(-2*ypval)) ## convert to logistic score  
      }
    }else{
      ypscore=yp
      if(isval){
        ypvalscore=ypval
      }
    }
    
  }
  
  ret=list()
  for(fam in families){
    if(!isval){
      bestScoreRound=iter
    }
    ret[[toString(fam)]] = BoostingModel(finalModel[[toString(fam)]][1:bestScoreRound],rate=v)
  }
  ret[["rate"]]=v
  ret[["log"]]=log
  ret[["bestScoreRound"]]=if(isval) bestScoreRound else iter
  ret[["rpartcontrols"]]=controls
  return(ret)  
}