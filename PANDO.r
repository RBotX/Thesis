source("helper.R")


TrainMultiTaskClassificationGradBoost = function(df,valdata=NULL,earlystopping=100,iter=3,v=1,groups,controls,ridge.lambda,target="binary",treeType="rpart",fitTreeCoef=FALSE,unbalanced=FALSE){
  scoreType = if(target == "binary") "auc" else "rmse"
  log=list()
  log[["tscore"]]=c()
  log[["vscore"]]=c()
  
  families = unique(groups)[unique(groups)!="clean"]
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
    #bestVscore=ypvalscore
    bestVscore=scorefunc(label=valdata$Label,preds=ypvalscore,scoreType=scoreType)
    
    
  }
  
  #numFamilies = length(unique(groups))-1 ## clean doesn't count as family
  numFamilies = length(unique(groups)) ## clean doesn't count as family
  finalModel[["rate"]]=v
  for(fam in families){
    finalModel[[toString(fam)]]=list()
    finalModel[[toString(fam)]][[1]] = preds
    
    
  }
  bestScoreRound=1
  for(t in 2:iter){
    
    if((isval)&(t-bestScoreRound > earlystopping)){
      cat("EARLY STOPPING AT ",t," best iteration was ",bestScoreRound," with validation score ",bestVscore,"\n")
      break
    }
    ## for each new tree, we have a new leaf->coef per family
    for(fam in families){
      leavesToCoefs = list()
      leavesToCoefs[[toString(fam)]]=list() 
    }
    tscore = scorefunc(label=data$Label,preds=yp,scoreType=scoreType)
    log[["tscore"]]=c(log[["tscore"]],tscore)
    if(isval){
      vscore = scorefunc(label=valdata$Label,preds=ypvalscore,scoreType=scoreType)
      if(((vscore > bestVscore)&(scoreType == "auc"))||((vscore < bestVscore)&(scoreType == "rmse"))){
        bestVscore = vscore
        
        bestScoreRound=t
      }
      log[["vscore"]]=c(log[["vscore"]],vscore)
      
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
    
    ## create a tree for all families together, 1 vs 0
    if(treeType=="rpart"){
      fit=rpart(pr~.,data=data[,-which(colnames(data) %in% c("Label","Family"))],control=controls,method="anova")  
      environment(fit$terms) <- NULL
      obsToLeaf = fit$where
      #fit = purge(fit)
    }else{
      fit = ctree(y~.,data=data.frame(x=data[,-which(colnames(data) %in% c("Label","Family"))],y=pr),control=controls,cores=3)
      obsToLeaf = predict(fit,data.frame(x=data[,-which(colnames(data) %in% c("Label","Family"))]),type="node")
    }
    
    
    
    ### fit model with per leaf score
    #leaves = unique(fit$where)
    leaves = unique(obsToLeaf)
    for(l in leaves){
      
      #samplesInLeaf = (fit$where==l) ## which are in the l-th leaf
      samplesInLeaf = (obsToLeaf==l) ## which are in the l-th leaf
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
      
      lambdas = 2^seq(3, -10, by = -.1)
      
      useglmnet=FALSE
      if(useglmnet){
        m=cv.glmnet(as.matrix(PerLeafData[,-which(colnames(PerLeafData) == "y")]),as.matrix(PerLeafData[,"y"]), alpha = 0, lambda = lambdas,intercept=FALSE,nfolds=3,standardize=T)  
        leafCoefs = coef(m,s="lambda.min")
        fittedFamilies = rownames(leafCoefs)[-1]
        leafCoefs = data.frame(as.matrix(leafCoefs))[-1,]
        leafCoefs = setNames(leafCoefs,fittedFamilies)
        
      }else{
        m = lm.ridge(y~.-1,data = PerLeafData,lambda=lambdas) 
        whichIsBest <- which.min(m$GCV) 
        leafCoefs=coef(m)[whichIsBest,]
        fittedFamilies = names(leafCoefs)
      }
      
      for(fam in families){
        if(fam %in% fittedFamilies){
          #intercept = as.numeric(coef(m)[1])
          #leavesToCoefs[[toString(fam)]][[l]]=as.numeric(leafCoefs[toString(fam)])
          intercept = 0
          
          leavesToCoefs[[toString(fam)]][[l]]=as.numeric(leafCoefs[fam])
          

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
      if(fitTreeCoef){
        famX = predict(finalModel[[toString(fam)]][[t]] ,data[(data[,"Family"]==fam),-which(colnames(data) %in% c("Label","Family"))])        
        famy = pr[(data[,"Family"]==fam)]
        lmdf = matrix(ncol=2,nrow=length(famX))
        lmdf[,1]=as.matrix(famX,ncol=1)
        lmdf[,2]=as.matrix(famy,ncol=1)
        colnames(lmdf)=c("x","y")
        lmdf = data.frame(lmdf)
        mm = lm(y~x -1,data=lmdf)
        fittedCoef = as.numeric(coef(mm)[1])
        if(is.na(fittedCoef)){
          fittedCoef=1
        }
        fittedIntercept = 0
        finalModel[[toString(fam)]][[t]] = TreeWithCoef(finalModel[[toString(fam)]][[t]],fittedCoef,fittedIntercept,treeType="rpart")
      }
      
    }
    
    famPreds=matrix(ncol=1,nrow=length(yp))
    if(isval){
      valfamPreds=matrix(ncol=1,nrow=length(ypval))      
    }
    
    for(fam in families){
      pp = predict(finalModel[[toString(fam)]][[t]],data[data[,"Family"]==fam,-which(colnames(data) %in% c("Label","Family"))])
      famPreds[data[,"Family"]==fam,1]=as.matrix(pp,ncol=1)
      if(isval){
        ppval = predict(finalModel[[toString(fam)]][[t]],valdata[valdata[,"Family"]==fam,-which(colnames(data) %in% c("Label","Family"))])  
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
  
  #return(finalModel)  
  ret=list()
  for(fam in families){
    if(!isval){
      bestScoreRound=iter
    }
    ret[[toString(fam)]] = BoostingModel(finalModel[[toString(fam)]][1:bestScoreRound],rate=rate)
  }
  ret[["rate"]]=v
  ret[["log"]]=log
  ret[["bestScoreRound"]]=if(isval) bestScoreRound else iter
  return(ret)  
  
}