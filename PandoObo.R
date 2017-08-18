source("helper.R")


## calculate the L12 penalty for the joint coefficient matrix: column j is the coefficient for covaraite j across tasks. 
L12penalty = function(coefMatrix){
  
  if(ncol(coefMatrix)==1){
    return(sqrt(sum(coefMatrix**2)))
  }
  
  return(sum(apply(coefMatrix,MARGIN = 2,FUN=function(x){sqrt(sum(x**2))})))
}

## minlambda - when reaching this lambda, stop- equivalent to number of iterations / early stopping mechanism
## 
## 
Obo = function(df,valdata=NULL,minlambda=0.0001,iter=10000,v=0.01,groups,controls,target="binary",df.val=NULL,treeType="rpart"){
  
  scoreType = if(target == "binary") "auc" else "rmse"
  lossFunc = if(target == "binary") negBinLogLikeLoss else squareLoss
  

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
    bestVscore=scorefunc(label=valdata$Label,preds=ypvalscore,scoreType=scoreType)
    
    
  }
  
  numFamilies = length(unique(groups)) 
  finalModel[["rate"]]=v
  for(fam in families){
    finalModel[[toString(fam)]]=list()
    finalModel[[toString(fam)]][[1]] = preds
    
  }
  bestScoreRound=1
  coefMatrix = matrix(ncol=0,nrow=numFamilies)
  #rownames(coefMatrix) = families
  
  
  lambdas = c()
  gamma_prev=0
  gamma_t=0
  for(t in 2:iter){
    
    
    tscore = scorefunc(label=data$Label,preds=yp,scoreType=scoreType)
    log[["tscore"]]=c(log[["tscore"]],tscore)
    if(!is.null(valdata)){
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
    
    
    pr = negative_gradient(y=data$Label,preds=ypscore,target=target) ## as if y-yp but multiply each adition by v so it's y-v*yp
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
    fittedCoefs = c()
    for(fam in families){
      ###  fit a coefficient per entire tree
      famx = data[(data[,"Family"]==fam),-which(colnames(data) %in% c("Label","Family"))]
      
      if(treeType=="rpart"){
        famX=predict(fit,famx)  
      }else{
        famX=predict(fit,data.frame(x=famx))  
      }
      
      famy = pr[(data[,"Family"]==fam)]
      #famy = data[data[,"Family"]==fam,"Label"] ## it says to take the derivative with respect to the loss function, but is it the loss function of pseudo responses or original responses?
      lmdf = matrix(ncol=2,nrow=length(famX))
      lmdf[,1]=as.matrix(famX,ncol=1)
      lmdf[,2]=as.matrix(famy,ncol=1)
      colnames(lmdf)=c("x","y")
      lmdf = data.frame(lmdf)
      fittedIntercept = 0
      fittedCoef = mean(negative_gradient(y=lmdf[,"y"],preds=lmdf[,"x"],target)) ## like in obozinski, coefficient is the negative gradient in the direction of the chosen covariate 
      fittedCoefs = c(fittedCoefs,fittedCoef)
      finalModel[[toString(fam)]][[t]] = TreeWithCoef(fit,fittedCoef,fittedIntercept,treeType=treeType)
      
    }
    coefMatrix = cbind(coefMatrix,matrix(fittedCoefs,ncol=1)) ## update coefficient matrix with coefs for the newly found tree
    
    ##### finished forward step: tree generation was ~ feature selection, coefficient (step size) is a negative gradient step
    ##### start backward steps loop:
    #####   a. choose which tree (~covaraite) across tasks to modify its coefficient
    #####   b. modify coefficient with some line search algorithm
    #####   c. update Gamma, stop when Gamma doesn't descrease significantly
    

    
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
    prev_ypscore = ypscore
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
    
    if(t==2){
      lambda_t = sum(lossFunc(preds=preds,y=data$Label))/L12penalty(coefMatrix)
    }else{
      
      nextLambda = (sum(lossFunc(preds=prev_ypscore,y=data$Label))- sum(lossFunc(preds=ypscore,y=data$Label)))/(L12penalty(coefMatrix) - L12penalty(matrix(coefMatrix[, -ncol(coefMatrix)],ncol=ncol(coefMatrix)-1)))#rate
      lambda_t = min(lambda_t, nextLambda)
    }
    cat("lambda_t is ",lambda_t," ", t, "\n")
    
    
    ### take a series of backward steps until Gamma doesn't decrease anymore
    gamma_prev = gamma_t
    gamma_t = lambda_t*L12penalty(coefMatrix) + sum(lossFunc(preds=ypscore,y=data$Label))
    while((min(gamma_t,gamma_prev)/max(gamma_t,gamma_prev)) < 0.99){ #continue until we improve by less of 1%
      print("gammas are: gamma_t:", gamma_t, " gamma_prev:", gamma_prev," ratio:",(min(gamma_t,gamma_prev)/max(gamma_t,gamma_prev)),"\n")
      ## choose covariate (tree) to modify its coefficient across tasks:
      gamma_grads=matrix(nrow=numFamilies,ncol=0) ## column k is the partial gradient with respect to w_k. one row per task
      for(j in 2:t){
        coefs = coefMatrix[,j-1] # the coefficient vector for covariate (tree) chosen at iteration j
        coefsnorm = sqrt(sum(coefs**2))
        loss_grads = c()
        for(fam in families){
          preds = predict(finalModel[[toString(fam)]][[j]],data[data[,"Family"]==fam,-which(colnames(data) %in% c("Label","Family"))])  
          labels = data[data[,"Family"]==fam,"Label"]
          loss_grad = negative_gradient(y = labels,preds=preds,target=tagrget)
          loss_grad = mean(loss_grad) # estimate gradient at this point
          loss_grads = c(loss_grads,loss_grad)
        }
        gamma_grad = lambda_t*(coefs/coefsnorm) - loss_grads # remember loss grads are negative gradient
        gamma_grads = cbind(gamma_grads,gamma_grad)
      }
      ## chosen covariate will be the column of gamma_grads whose norm2 is the largest
      gammaNorms = apply(gamma_grads,MARGIN = 2,function(x){sqrt(sum(x**2))})
      chosenCovariateIndex = which(gammaNorms == max(gammaNorms))
      cat("backward step: fixing covariate ",chosenCovariateIndex,"\n")
      ## now we have the chosen covariate, modify its coefficient accordingly with a backward step:
      coefs = coefMatrix[,chosenCovariateIndex]
      coefsnorm = sqrt(sum(coefs**2))
      coefs = coefs - lambda_t*(coefs/coefsnorm) - loss_grads # step size for backward step as described in short version --> is this line search equivalent?
      coefMatrix[,chosenCovariateIndex] = coefs ## update coefficient matrix
      ## update the relevant cofficient per task:
      for(fam in families){
        finalModel[[toString(fam)]][[chosenCovariateIndex+1]]$fittedCoef = coefMatrix[which(families==fam),chosenCovariateIndex] 
      }
      gamma_prev = gamma_t
      gamma_t = lambda_t*L12penalty(coefMatrix) + sum(lossFunc(preds=ypscore,y=data$Label))
      ### updated gamma, finish this backward step, find next covariate to fix
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
  return(ret)  
}