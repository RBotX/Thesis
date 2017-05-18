source("helper.R")
TrainMultiTaskClassificationGradBoost = function(df,valdata=NULL,iter=3,v=1,groups,controls,ridge.lambda,target="binary",treeType="rpart"){
  families = unique(groups)[unique(groups)!="clean"]
  data = df  
  finalModel=list()
  isval=!is.null(valdata)
  if(target=="binary"){
    preds = 0#0.5 * log( (1+mean(data$Label))/(1-mean(data$Label))  ) ### initial model  
  }else{
    preds = 0#mean(data$Label)
  }
  
  finalModel[[1]]=preds
  yp = rep(preds,nrow(data)) ## initial guess without learning rate?
  ypscore=yp
  if(isval){
    ypval = rep(preds,nrow(valdata)) ## initial guess without learning rate?
    ypvalscore= ypval
    
  }
  
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
        cat("train AUC is:",roc(as.factor(data$Label),as.numeric(ypscore))$auc[1],"\n") 
        if(!is.null(valdata)){
          cat("val AUC is:",roc(as.factor(valdata$Label),as.numeric(ypvalscore))$auc[1],"\n")           
        }
      }
    }
    
    #cat(head(yp,n=50),"-----------\n")
    pr = negative_gradient(y=data$Label,preds=ypscore,target=target) ## as if y-yp but multiply each adition by v so it's y-v*yp
    
    ## create a tree for all families together, 1 vs 0
    if(treeType=="rpart"){
      fit=rpart(pr~.,data=data[,-which(colnames(data) %in% c("Label","Family"))],control=controls,method="anova")  
    }else{
      fit = ctree(y~.,data=data.frame(x=data[,-which(colnames(data) %in% c("Label","Family"))],y=pr),control=controls,cores=3)
    }
    
    
    
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
    valfamPreds=matrix(ncol=1,nrow=length(ypval))
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
    }
    
    
  }
  #return(finalModel)  
  ret=list()
  for(fam in families){
    ret[[toString(fam)]] = BoostingModel(finalModel[[toString(fam)]],rate=rate)
  }
  ret[["rate"]]=v
  return(ret)  
  
}