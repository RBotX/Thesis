source("helper.R")


## calculate the L12 penalty for the joint coefficient matrix: column j is the coefficient for covaraite j across tasks. 
L12penalty = function(coefMatrix){
  
  if(ncol(coefMatrix)==1){
    return(sqrt(sum(coefMatrix**2)))
  }
  
  return(sum(apply(coefMatrix,MARGIN = 2,FUN=function(x){sqrt(sum(x**2))})))
}


## for given tree, do line search to minimize OBJECTIVE (gamma)
optim_lossFunc = function(data,par){
  
  preds = data$preds
  labels = data$Label
  groups = data$groups
  target = data$target
  lambda = data$lambda
  lossFunc = data$lossFunc
  chosenCovariateIndex=data$chosenCovariateIndex
  coefMatrix = data$coefMatrix
  families = unique(data$groups) 
  treePreds = data$treePreds
  newTreePreds = rep(0,length(treePreds))
  for(fam in families){
    newTreePreds[groups==fam] = (par[toString(fam)])*treePreds[groups==fam]
  }
  newpreds = preds + newTreePreds
  if(target=="binary"){
    newpreds = 1/(1+exp(-2*newpreds)) ## convert to logistic score 
  }
  newcoefs=c()
  for(fam in families){ ## update coefficient matrix to match new found coefs
    newcoefs = c(newcoefs,par[[toString(fam)]])
  }
  newcoefMatrix=coefMatrix
  newcoefMatrix[,chosenCovariateIndex]=matrix(newcoefs,ncol=1)
  
  ret=lambda*L12penalty(newcoefMatrix) + mean(lossFunc(preds=newpreds,y=data$Label)) ### return new value of *objective*
  #cat("L12 is: ",L12penalty(newcoefMatrix), " loss is: ",mean(lossFunc(preds=newpreds,y=data$Label)),"\n")
  #write.csv(treePreds,"treePreds.csv")
  return(ret)
}


## minlambda - when reaching this lambda, stop- equivalent to number of iterations / early stopping mechanism
## 
## 
Obo = function(df,valdata=NULL,earlystopping=100,iter=10000,v=0.01,groups,controls,target="binary",df.val=NULL,treeType="rpart"){
  earlystopping=100
  scoreType = if(target == "binary") "auc" else "rmse"
  lossFunc = if(target == "binary") negBinLogLikeLoss else squareLoss
  lambdaNegative=FALSE

  log=list()
  log[["tscore"]]=c()
  log[["vscore"]]=c()
  log[["tloss"]]=c()
  log[["vloss"]]=c()
  families = unique(groups)[unique(groups)!="clean"]
  data = df  
  finalModel=list()
  isval=!is.null(valdata)
  if(target=="binary"){
    preds = 0.5 * log( (1+mean(data$Label))/(1-mean(data$Label))  ) ### initial model  
  }else{
    preds = mean(data$Label)
  }
  #preds = 0*preds
  finalModel[[1]]=preds
  yp = rep(preds,nrow(data)) ## initial guess without learning rate?
  ypscore=v*yp
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
  lambda_t=0
  for(t in 2:iter){
    cat("starting iteartion ",t,"\n\n")
    if((isval)&(t-bestScoreRound > earlystopping)){
      cat("EARLY STOPPING AT ",t," best iteration was ",bestScoreRound," with validation score ",bestVscore,"\n")
      break
    }
    
    tscore = scorefunc(label=data$Label,preds=ypscore,scoreType=scoreType)
    tloss = lossFunc(preds=ypscore,y=data$Label)
    log[["tscore"]]=c(log[["tscore"]],tscore)
    log[["tloss"]]=c(log[["tloss"]],tloss)
    if(!is.null(valdata)){
      vscore = scorefunc(label=valdata$Label,preds=ypvalscore,scoreType=scoreType)
      if(((vscore > bestVscore)&(scoreType == "auc"))||((vscore < bestVscore)&(scoreType == "rmse"))){
        bestVscore = vscore
        bestScoreRound=t
      }
      
      log[["vscore"]]=c(log[["vscore"]],vscore)
    }
    
    
    ### pseudo responses
    if(t%%20 >= 0){
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
      if(lambda_t<0){
        cat("bla bla\n")
      }
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
      
      famy = pr[(data[,"Family"]==fam)] ## this isn't labels... 
      #famy = data[data[,"Family"]==fam,"Label"] ## it says to take the derivative with respect to the loss function, but is it the loss function of pseudo responses or original responses?
      lmdf = matrix(ncol=2,nrow=length(famX))
      lmdf[,1]=as.matrix(famX,ncol=1)
      lmdf[,2]=as.matrix(famy,ncol=1)
      colnames(lmdf)=c("x","y")
      lmdf = data.frame(lmdf)
      fittedIntercept = 0
      ## TODO: fix this, I think this is wrong --> maybe the gradient should be that of "regression" as that's how we fit in rpart?
      fittedCoef = v*mean(negative_gradient(y=lmdf[,"y"],preds=lmdf[,"x"],target)) ## like in obozinski, coefficient is a negative epsilon gradient step  
      fittedCoefs = c(fittedCoefs,fittedCoef)
      finalModel[[toString(fam)]][[t]] = TreeWithCoef(fit,fittedCoef,0,treeType=treeType)
      
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
    #yp = yp + v*famPreds
    yp = yp + famPreds ## in this flavor we encode the rate into the coefficeint of the tree
    if(isval){
      #ypval = ypval + v*valfamPreds
      ypval = ypval + valfamPreds # in this flavor we encode the rate into the coefficient of the tree
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
      lambda_t = mean(lossFunc(preds=preds,y=data$Label))/L12penalty(coefMatrix)
    }else{
      nextLambda = (mean(lossFunc(preds=prev_ypscore,y=data$Label)) - mean(lossFunc(preds=ypscore,y=data$Label)))/(L12penalty(coefMatrix) - L12penalty(matrix(coefMatrix[, -ncol(coefMatrix)],ncol=ncol(coefMatrix)-1)))#rate
      if(nextLambda>0){
        lambda_t = min(lambda_t, nextLambda)
        lambdaNegative=FALSE
      }else{
        lambdaNegative=TRUE
      }
    }
    #cat("mean loss is: ",mean(lossFunc(preds=prev_ypscore,y=data$Label))," lambda_t is: ",lambda_t,"\n")
    
    
    ### take a series of backward steps until Gamma doesn't decrease anymore
    gamma_prev = gamma_t
    gamma_t = lambda_t*L12penalty(coefMatrix) + mean(lossFunc(preds=ypscore,y=data$Label)) ## current gamma
    

    cat("gammas are: gamma_t:", gamma_t, " gamma_prev:", gamma_prev," distance:",abs(gamma_t-gamma_prev),"\n")
    if(lambdaNegative){
      cat("loss is: ",mean(lossFunc(preds=prev_ypscore,y=data$Label))," negative lambda value, contiuing\n")
      next
    }
    if(t<10) next
    correctionStepsTaken=0
    #while(abs(gamma_t-gamma_prev) > 1e-06){
    for(covarIndex in 1:ncol(coefMatrix)){
      correctionStepsTaken = correctionStepsTaken+1
      cat("iteration: ",t, " lambda_t: ",lambda_t," gammas are: gamma_t:", gamma_t, " gamma_prev:", gamma_prev," distance:",abs(gamma_t-gamma_prev),"\n")
      ## choose covariate (tree) to modify its coefficient across tasks:
      # gamma_grads=matrix(nrow=numFamilies,ncol=0) ## column k is the partial gradient with respect to w_k. one row per task
      # for(j in 2:t){
      #   coefs = coefMatrix[,j-1] # the coefficient vector for covariate (tree) chosen at iteration j
      #   coefsnorm = sqrt(sum(coefs**2))
      #   loss_grads = c()
      #   for(fam in families){
      #     preds = predict(finalModel[[toString(fam)]][[j]],data[data[,"Family"]==fam,-which(colnames(data) %in% c("Label","Family"))])
      #     labels = data[data[,"Family"]==fam,"Label"]
      #     if(target=="binary"){ ## calibrate predictions if binary
      #       preds = 1/(1+exp(-2*preds)) ## convert to logistic score
      #     }
      #     loss_grad = negative_gradient(y = labels,preds=preds,target=target)/(length(yp))
      #     loss_grad = mean(loss_grad) # estimate gradient at this point
      #     loss_grads = c(loss_grads,loss_grad)
      #   }
      #   loss_grads = -loss_grads # we calculated the negative gradient
      #   gamma_grad = lambda_t*(coefs/coefsnorm) + loss_grads 
      #   gamma_grads = cbind(gamma_grads,gamma_grad)
      # }
      # ## chosen covariate will be the column of gamma_grads whose norm2 is the largest
      # gammaNorms = apply(gamma_grads,MARGIN = 2,function(x){sqrt(sum(x**2))})
      # cat("gamma norms is: ",gammaNorms,"\n")
      # gammaNorms[is.nan(gammaNorms)]=-1 ### if coefficient was zeroed by previous correction step, ignore this covariate
      # chosenCovariateIndex = as.numeric(min(which(gammaNorms == max(gammaNorms))))
      # if(chosenCovariateIndex>1){
      #   cat("finally another covariate!\n")
      # }
      chosenCovariateIndex = covarIndex
      #cat("backward step: fixing covariate ",chosenCovariateIndex+1,"\n")
      ## now we have the chosen covariate, modify its coefficient accordingly with a backward step:
      coefs = coefMatrix[,chosenCovariateIndex]
      coefsnorm = sqrt(sum(coefs**2))
      #newcoefs = coefs -  v*((lambda_t*(coefs/coefsnorm)) + loss_grads) # step size for backward step as described in short version --> is this line search equivalent?
      
      
      ######################
      dat=list()
      dat[["Label"]]=data$Label
      dat[["gamma_t"]]=gamma_t
      dat[["lambda"]] = lambda_t
      dat[["coefMatrix"]] = coefMatrix
      dat[["target"]] = target
      dat[["chosenCovariateIndex"]] = chosenCovariateIndex
      dat[["lossFunc"]]=lossFunc
      dat[["finalModel"]]=finalModel
      
      par=list()
      treePreds = matrix(ncol=1,nrow=length(yp))
      fullPreds = matrix(ncol=1,nrow=length(yp))
      for(fam in families){
         fullFamPreds=predict(finalModel[[toString(fam)]][[chosenCovariateIndex+1]],data[data[,"Family"]==fam,-which(colnames(data) %in% c("Label","Family"))]) ## predict with cofficient per task
         treeFamPreds=predict(finalModel[[toString(fam)]][[chosenCovariateIndex+1]]$treeFit ## predict using the fitted tree object
                              ,data[data[,"Family"]==fam,-which(colnames(data) %in% c("Label","Family"))]) ## only raw tree prediction
         fullPreds[data[,"Family"]==fam,]=fullFamPreds
         treePreds[data[,"Family"]==fam,]=treeFamPreds
         par[[toString(fam)]]=0
      }
      dat[["treePreds"]] = treePreds # raw tree prediction
      dat[["preds"]] = yp-fullPreds # raw tree prediction
      dat[["groups"]] = data[,"Family"]
      #optimres = optim(par, optim_lossFunc,data=dat,method = "BFGS") --> this causes repeteadly to select the first three, and it almost zeroes it but never actually does
      optimres = optim(par, optim_lossFunc,data=dat) ## default method is Nelder and Mead (1965), "Nelder-Mead", apparently this zeroes the covariate instead of dangling around zero
      if(optimres$value==gamma_t){
        cat("optim couldn't reduce Gamma, continuing to next coefficient\n")
        next
      }
      newcoefs = matrix(optimres$par,ncol=1) 
      if( (optimres$value-gamma_t) > 0 ){
        cat("****************optim didn't improve gamma\n")
      }else{
        cat("****************optim improved  gamma\n")
      }
      ######################
      cat("replacing coefficients: ",chosenCovariateIndex+1,"\n",coefMatrix[,chosenCovariateIndex], "\n", newcoefs,"\n")
      if(any(newcoefs > 0.1)){
        cat("taking a big step, lambda now is: ",lambda_t,"\n")
        
      }
      coefMatrix[,chosenCovariateIndex] =  v*newcoefs ## update coefficient matrix
      
      
      ## update predictions
      yp = yp-fullPreds
      newTreePreds = rep(0,length(yp))
      for(fam in families){
        finalModel[[toString(fam)]][[chosenCovariateIndex+1]]$fittedCoef = coefMatrix[which(families==fam),chosenCovariateIndex] 
        yp[data[,"Family"]==fam] =  yp[data[,"Family"]==fam] + predict(finalModel[[toString(fam)]][[chosenCovariateIndex+1]],data[data[,"Family"]==fam,-which(colnames(data) %in% c("Label","Family"))])
        ## update *exactly* like in optim
        #yp[data[,"Family"]==fam] =  yp[data[,"Family"]==fam] + coefMatrix[which(families==fam),chosenCovariateIndex]*treePreds[data[,"Family"]==fam] 
        
        
        #newTreePreds[data[,"Family"]==fam] = predict(finalModel[[toString(fam)]][[chosenCovariateIndex+1]],data[data[,"Family"]==fam,-which(colnames(data) %in% c("Label","Family"))])
        #yp[data[,"Family"]==fam] =  yp[data[,"Family"]==fam]+predict(finalModel[[toString(fam)]][[chosenCovariateIndex+1]],data[data[,"Family"]==fam,-which(colnames(data) %in% c("Label","Family"))])
      }
      if(target=="binary"){ ## calibrate predictions if binary
        ypscore = 1/(1+exp(-2*yp)) ## convert to logistic score
        if(isval){
          ypvalscore = 1/(1+exp(-2*ypval)) ## convert to logistic score
        }
      }
      
      
      ## update predictions - i think this important, as we want to update the gradient of the *objective* with respect to the  *updated* model, otherwise
      ## we won't budge from the prevoiusly chosen covariate 
      gamma_prev = gamma_t
      
      gamma_t = lambda_t*L12penalty(coefMatrix) + mean(lossFunc(preds=ypscore,y=data$Label))
      
      gamma_change=gamma_t-gamma_prev
      cat("after replacing coefficients change  gamma is: ",gamma_t,"previous gamma is: ",gamma_prev,"\n") ## should be negative as we want gamma to decrease!
      cat("after replacing coefficients loss is: ",mean(lossFunc(preds=ypscore,y=data$Label)),"\n") ## should be negative as we want gamma to decrease!
      if(round(gamma_t - optimres$value,7)!=0){
        cat("why\n")
      }
      if(gamma_change > 0){
        cat("gamma not reduced\n")
      }
      cat("finished backwatd step")
    }
    
  }
  
  ret=list()
  for(fam in families){
    if(!isval){
      bestScoreRound=iter
    }
    ret[[toString(fam)]] = BoostingModel(finalModel[[toString(fam)]][1:bestScoreRound],rate=1) ## in this flavor we will enode the rate in the coefficient of each tree
  }
  ret[["log"]]=log
  ret[["bestScoreRound"]]=if(isval) bestScoreRound else iter
  return(ret)  
}