source("helper.R")

getLeafScores2 = function(treefit,groups,pr){
  families=unique(groups)
  numFamilies = length(families)
  leaves = unique(treefit$where)
  obsToLeaf = treefit$where
  leavesToCoefs = list()
  for(l in leaves){
    samplesInLeaf = (obsToLeaf==l) ## which are in the l-th leaf
    if(length(which(samplesInLeaf))==0){
      stop("zero samples - bad!\n")

    }
    
    
    ### take leaf predictions in that leaf, treat those as a variable in a ridge prediction
    ridgeRegX = matrix(nrow=0,ncol=numFamilies)
    ridgeRegy = matrix(nrow=0,ncol=1)
    PadColsToLeft = 0
    PadColsToRight = numFamilies-1
    famIndicator = matrix(ncol=1,nrow=0)
    oboData=c()
    for(fam in families){
      ## check if this family has results in this leaf, if not, skip
      familiesInLeaf = length(unique(groups[(samplesInLeaf)]))
      dataInFamInLeafCount=length(which((samplesInLeaf)&(groups==fam)))
      #thisLeafPrediction =  as.numeric(unique(newLearnerPredictions[(samplesInLeaf)]))
      if(dataInFamInLeafCount == 0){
        next
      }
      
      
      y = pr[(samplesInLeaf)&(groups==fam)]
      
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
      
      for(fam in families){
        leavesToCoefs[[toString(fam)]][[toString(l)]]=mean(y)
      }
      next
    }
    
    PerLeafData = data.frame(ridgeRegX)
    PerLeafData = cbind(PerLeafData,y)
    #cat("before ridge target is ",head(y),"***********\n")
    # if(fitCoef=='ridge'){
    lambdas = 2^seq(6, -10, by = -1)
    useglmnet=FALSE
    
    if(useglmnet){
      m=cv.glmnet(as.matrix(PerLeafData[,-which(colnames(PerLeafData) == "y")]),as.matrix(PerLeafData[,"y"]), alpha = 0, nlambda = 100,intercept=FALSE,nfolds=3,standardize=FALSE)  
      leafCoefs = coef(m,s="lambda.min")
      fittedFamilies = rownames(leafCoefs)[-1]
      leafCoefs = as.data.frame(as.matrix(leafCoefs))[-1,]
      names(leafCoefs) = fittedFamilies
      
    }else{
      m = lm.ridge(y~.-1,data = PerLeafData,lambda=lambdas) 
      whichIsBest <- which.min(m$GCV) 
      leafCoefs=coef(m)[whichIsBest,]
      fittedFamilies = names(leafCoefs)
    }
    # }

    for(fam in families){
          intercept = 0
          if(!(fam %in% fittedFamilies)){
            leafCoefs=c(leafCoefs,mean(y))
            names(leafCoefs)[length(names(leafCoefs))]=fam
          }
          leavesToCoefs[[toString(fam)]][[toString(l)]]=as.numeric(leafCoefs[fam])  
          if(is.na(as.numeric(leafCoefs[fam])) | is.na(mean(y))){
            cat("hey")
          }
          
    }
    
    
  }
  for(fam in families){
    
    leavesToCoefs[[toString(fam)]] =   leavesToCoefs[[toString(fam)]][-which(is.na(leavesToCoefs[[toString(fam)]]))]
  }
  return(leavesToCoefs)
}






####################################################3

getLeafScores = function(treefit,groups,pr){
  families=unique(groups)
  leaves = unique(treefit$where)
  obsToLeaf = treefit$where
  leafScores = list()
  for(l in leaves){
    samplesInLeaf = (obsToLeaf==l) ## which are in the l-th leaf
    if(length(which(samplesInLeaf))==0){
      next
    }
    leafScore=0
    for(fam in families){
      ## check if this family has results in this leaf, if not, skip
      #cat(fam," ",length(which((samplesInLeaf)&(data[,"Family"]==fam))) == 0,"\n")
      dataInFamInLeafCount=length(which((samplesInLeaf)&(groups==fam)))
      if(dataInFamInLeafCount == 0){
        next
      }
      
      y = pr[(samplesInLeaf)&(groups==fam)]
      leafScore = leafScore + mean(y)
    }
    leafScores[[toString(l)]]=leafScore
  }
  return(leafScores)
}

#########################################################
getLeafScores3 = function(treefit,groups,pr){
  families=unique(groups)
  leaves = unique(treefit$where)
  obsToLeaf = treefit$where
  leafScores = list()
  for(l in leaves){
    samplesInLeaf = (obsToLeaf==l) ## which are in the l-th leaf
    if(length(which(samplesInLeaf))==0){
      next
    }
    leafScore=0
    for(fam in families){
      ## check if this family has results in this leaf, if not, skip
      #cat(fam," ",length(which((samplesInLeaf)&(data[,"Family"]==fam))) == 0,"\n")
      dataInFamInLeafCount=length(which((samplesInLeaf)&(groups==fam)))
      if(dataInFamInLeafCount == 0){
        next
      }
      
      y = pr[(samplesInLeaf)&(groups==fam)]
      leafScore = leafScore + mean(y)
    }
    leafScores[[toString(l)]]=leafScore
  }
  return(leafScores)
}
###########################################################


editRpartRegressionTree = function(treeFit,leafScores){
  leafRowIndex = row.names(treeFit$frame)
  for(leafId in names(leafScores)){ 
    ## leafId is the row number in treeFit$frame that this leaf appears in: verified by  
    ## all(sort(unique(treeFit$where))==which(treeFit$frame[,"var"]=="<leaf>"))
    treeFit$frame[as.numeric(leafId),"yval"]=leafScores[[leafId]]
  }
  return(treeFit)
}

addTreeCoefRpart = function(treeFit,treeCoef){
  treeFit$frame[which(treeFit$frame[,"var"]=="<leaf>"),"yval"] = treeCoef*treeFit$frame[which(treeFit$frame[,"var"]=="<leaf>"),"yval"]
  return(treeFit)
}


### fit a coefficient per tree per task. all tasks share the same trees with different coefficients 
TrainMultiTaskClassificationGradBoost3 = function(df,valdata=NULL,earlystopping=100,iter=3,v=1,groups,controls,target="binary",fitLeafCoef="ridge",fitTreeCoef="ridge",treeType="rpart",unbalanced=FALSE){
  rate=v
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
      vscore=0
      for(fam in families){
        idxs=(valdata[,"Family"]==fam)
        vscore =vscore+ scorefunc(label=valdata$Label[idxs],preds=ypvalscore[idxs],scoreType=scoreType)  
      }
      vscore = vscore/length(families) ## average scores
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
    
    
    
    
    
    ##### building obo leaf scores:
    familyTree=list()
    if(fitLeafCoef=="ridge" | fitLeafCoef=="nocoef"){
      leavesToCoefs = getLeafScores2(treefit=fit,groups=groups,pr=pr)
      for(fam in families){
        familyTree[[fam]]=editRpartRegressionTree(fit,leavesToCoefs[[fam]])
      }
    }
    if(fitLeafCoef=="obo"){
     leavesToCoefs = getLeafScores(treefit=fit,groups=groups,pr=pr)
     for(fam in families){
       familyTree[[fam]]=editRpartRegressionTree(fit,leavesToCoefs)
     }
    }
    # if(fitLeafCoef=="nocoef"){ ## to be used in pando with tree coef, and vanillaboost
    #   for(fam in families){
    #     familyTree[[fam]]=fit
    #   }
    #}
    
    
    
    ### editing model per tree
    
    #fit=editRpartRegressionTree(fit,leavesToCoefs) 
    ## now are tree model includes leaf scores which reflect l12 reg as they minimize
    ## block wise loss instead of overall loss
    
    
    ##### fitting the coefficient per tree
    ridgeRegX = NULL
    ridgeRegy = NULL
    if(fitTreeCoef=="ridge" | fitTreeCoef=="obo"){
      for(fam in families){
        ###  fit a coefficient per entire tree per family
        fit=familyTree[[fam]] ## get tree for this family, regrdless of the used method
        
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
      
      if(fitTreeCoef=='ridge')
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
    if(fitTreeCoef == "obo"){        
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
      #cat("obo coefs are: ",obocoefs,"\n\n")
    }
    
    for(fam in families){
      if(fitTreeCoef == "obo"){
        fittedCoef=as.numeric(obocoefs[fam])
      }
      
      if(fitTreeCoef == "nocoef"){
        fittedCoef=1 ## standard boosting
      }
      
      if(fitTreeCoef == "ridge"){
        fittedCoef = as.numeric(coefs[fam])          
      }
      
      fittedIntercept = 0
      finalModel[[toString(fam)]][[t]] = TreeWithCoef(familyTree[[fam]],fittedCoef,fittedIntercept,treeType=treeType)
      
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
    ret[[toString(fam)]] = BoostingModel(finalModel[[toString(fam)]][1:bestScoreRound],rate=rate)
  }
  ret[["log"]]=log
  ret[["bestScoreRound"]]=if(isval) bestScoreRound else iter
  ret[["rpartcontrols"]]=controls
  return(ret)  
}