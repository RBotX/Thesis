setwd("/home/dan/Thesis")
source("helper.R")
source("PANDO.r")
source("PANDO2.r")
library(MASS)
library(grpreg)
library(xtable)
library(knitr)


## http://www.jmlr.org/papers/volume8/xue07a/xue07a.pdf
## families 1-15 are from highliy foliated regions
## families 16-29 are from earth/desert regions
schools2=read.csv("LandmineData.csv",stringsAsFactors = FALSE)
schools2=schools2[,-1]
schools2[schools2[,"Label"]==0,"Label"]=-1
schools2=schools2[schools2[,"Family"] %in% paste0("fam",0:15),]




alltests=c()
NUM_TESTS=1
for(l in 1:NUM_TESTS){
  set.seed(l)
  testidx = c()
  for(fam in unique(schools2[,"Family"])){
    
    testidx = c(testidx, sample(which(schools2[,"Family"]==fam),floor(length(which(schools2[,"Family"]==fam))*0.3) )) 
  }
  
  
  
  data=list()
  data[["data"]]=schools2
  data[["testidx"]]=testidx
  
  #controls=rpart.control(maxdepth = 3)
  iter=2000
  rate=0.005
  ridge.lambda=1  
  #data=GenerateData(d=d,ntrain=ntrain,ntest=ntest,seed=i)
  
  train = data$data[-data$testidx,]
  test = data$data[data$testidx,]
  
  mshared=TrainMultiTaskClassificationGradBoost(train,iter=iter,v=rate,groups=train[,"Family"],controls=rpart.control(maxdepth = 4, cp=0.0001),ridge.lambda=ridge.lambda,target="binary",valdata=test)
  mshared2=TrainMultiTaskClassificationGradBoost2(train,iter=iter,v=rate,groups=train[,"Family"],controls=rpart.control(maxdepth = 3, cp=0.0001),ridge.lambda=ridge.lambda,target="binary",treeType="rpart",valdata=test)
  #mshared3=TrainMultiTaskClassificationGradBoost2(train,iter=iter,v=rate,groups=train[,"Family"],controls=ctree_control(maxdepth = 4),ridge.lambda=ridge.lambda,target="regression",treeType="ctree",fitCoef="norm2")
  #mshared4=TrainMultiTaskClassificationGradBoost2(train,iter=iter,v=rate,groups=train[,"Family"],controls=ctree_control(maxdepth = 4),ridge.lambda=1,target="regression",treeType="ctree",fitCoef="norm1")
  
  perTaskModels=list()
  logitModels=list()
  for(fam in unique(train[,"Family"])){
    
    cat("fam ",fam,"\n")
    tr = train[train[,"Family"]==fam,]
    m0 = TrainMultiTaskClassificationGradBoost(tr,groups = matrix(fam,nrow=nrow(tr),ncol=1),iter=100,v=rate,
                                               controls=rpart.control(maxdepth = 4), ridge.lambda = ridge.lambda,target="binary")  
    perTaskModels[[toString(fam)]]=m0
    logitModels[[toString(fam)]]= cv.glmnet(x=as.matrix(tr[,-which(colnames(tr) %in% c("Family","Label"))]),y=tr[,"Label"],family="binomial",alpha=1,maxit=10000,nfolds=3, thresh=1E-4)
  }
  
  ### train binary model, ignoring multi tasking:
  binaryData = train
  binaryData["Family"]=1
  mbinary=TrainMultiTaskClassificationGradBoost(binaryData,iter=iter,v=rate,groups=matrix(1,nrow=nrow(binaryData),ncol=1),controls=rpart.control(maxdepth = 4),ridge.lambda=ridge.lambda,target="binary")  
  mlogitbinary = cv.glmnet(x=as.matrix(binaryData[,-which(colnames(tr) %in% c("Family","Label"))]),y=binaryData[,"Label"],family="binomial",alpha=1,maxit=10000,nfolds=3, thresh=1E-4)
  
  gplassotraindata = CreateGroupLassoDesignMatrix(train)
  gplassoX = (gplassotraindata$X)[,-ncol(gplassotraindata$X)]
  gplassoy =  (gplassotraindata$X)[,ncol(gplassotraindata$X)]
  gplassoy[gplassoy==-1]=0
  
  mgplasso = cv.grpreg(gplassoX, gplassoy, group=gplassotraindata$groups, nfolds=2, maxit=10000,seed=777,family="binomial",trace=TRUE)
  
  gplassotestdata = CreateGroupLassoDesignMatrix(test)
  gplassotestX = (gplassotestdata$X)[,-ncol(gplassotestdata$X)]
  gplassotesty =  (gplassotestdata$X)[,ncol(gplassotestdata$X)]
  
  gplassoPreds = predict(mgplasso,gplassotestX,type="response",lambda=mgplasso$lambda.min)
  
  methods = c("PANDO","PTB","BB","PTLogit","BinaryLogit","PANDO2","GL")#"PANDO3","GL","PANDO4")
  
  rc=list()
  tt=list()
  compmat = c()
  digitsfmt = matrix(-2,nrow=length(methods),ncol=length(methods))
  xtables=list()
  k=0
  allpreds = matrix(nrow=nrow(test),ncol=length(methods)+2)
  colnames(allpreds)=c(methods,"Label","testnum")
  allpreds[,"Label"]=test[,"Label"]
  allpreds[,"testnum"]=l
  ##################### test:
  for(fam in unique(test[,"Family"])){
    k = k+1
    testidxs = which(test["Family"]==fam)
    compmatrix = matrix(nrow=length(methods),ncol = length(methods))
    
    tr.test = test[test["Family"]==fam,]
    tr.test = tr.test[,-which(colnames(tr.test)=="Family")]
    
    tt[[methods[which(methods=="PANDO")]]]= predict(mshared[[toString(fam)]],tr.test[,-which(colnames(tr.test) %in% c("Family","Label"))],calibrate=TRUE)
    tt[[methods[which(methods=="PTB")]]]= predict(perTaskModels[[toString(fam)]][[toString(fam)]],tr.test[,-which(colnames(tr.test) %in% c("Family","Label"))],calibrate=TRUE)
    tt[[methods[which(methods=="BB")]]] = predict(mbinary[[toString(1)]],tr.test[,-which(colnames(tr.test) %in% c("Family","Label"))],calibrate=TRUE)
    tt[[methods[which(methods=="PTLogit")]]] =predict(logitModels[[toString(fam)]],newx=as.matrix(tr.test[,-which(colnames(tr) %in% c("Family","Label"))]),type="response",s=logitModels[[toString(fam)]]$lambda.min)
    tt[[methods[which(methods=="BinaryLogit")]]] =predict(mlogitbinary,newx=as.matrix(tr.test[,-which(colnames(tr) %in% c("Family","Label"))]),type="response",s=mlogitbinary$lambda.min)
    tt[[methods[which(methods=="PANDO2")]]] = predict(mshared2[[toString(fam)]],tr.test[,-which(colnames(tr.test) %in% c("Family","Label"))],calibrate=TRUE)
    #tt[[methods[7]]] = predict(mshared3[[toString(fam)]],tr.test[,-which(colnames(tr.test) %in% c("Family","Label"))],calibrate=TRUE)
    tt[[methods[which(methods=="GL")]]] = gplassoPreds[test[,"Family"]==fam]
    #tt[[methods[9]]] = predict(mshared4[[toString(fam)]],tr.test[,-which(colnames(tr.test) %in% c("Family","Label"))],calibrate=TRUE)
    
    
    
    
    for(i in 1:length(methods)){
      rc[[methods[i]]]=pROC::roc(as.factor(tr.test[,"Label"]),as.numeric(tt[[methods[i]]]))
      allpreds[testidxs,methods[i]] = as.matrix(tt[[methods[i]]],ncol=1)
    }
    
    for(i in 1:length(methods)){
      compmatrix[i,i]=rc[[methods[i]]]$auc[1]
      digitsfmt[i,i]=3
      for(j in 1:length(methods)){
        if(i >=j ){
          next
        }
        compmatrix[i,j] = 0#signif(pROC::roc.test(rc[[methods[i]]],rc[[methods[j]]])$p.value, digits = 3)
        cat("auc  for ",fam," ", methods[i]," VS ",methods[j],": ",round((rc[[methods[i]]]$auc[1]),4)," ",round((rc[[methods[j]]]$auc[1]),4),"\n")
      }
    }
    compmat = rbind(compmat,compmatrix)
    cat("***********\n")
    
    
    dft=data.frame(compmatrix)
    colnames(dft)=methods
    rownames(dft)=methods
    xtables[[toString(k)]]=xtable(dft,digits=cbind(rep(1,nrow(digitsfmt)),digitsfmt))
    
  }
  #####
  
  alltests=rbind(alltests,allpreds)
  cat("round ",l," summary:\n********************\n")
  for(method in methods){
    score=pROC::roc(as.factor(alltests[alltests[,"testnum"]==l,"Label"]),alltests[alltests[,"testnum"]==l,method])$auc[1]
    cat(method," ",score,"\n")
  }
  
  
  
}

cat("\n*********************************************\n")
cat("final results:\n")
finalresults= matrix(nrow=NUM_TESTS,ncol=length(methods))
#finalAUC= matrix(nrow=NUM_TESTS,ncol=length(methods))
colnames(finalresults)=methods
colnames(finalAUC)=methods
for(l in 1:NUM_TESTS){
  for(method in methods){
    
    score=pROC::roc(as.factor(alltests[alltests[,"testnum"]==l,"Label"]),alltests[alltests[,"testnum"]==l,method])
    cat(method," ",score$auc[1],"\n")
    finalresults[l,method]=score$auc[1]
    #finalAUC[l,method]=score
  }
}
for(m in colnames(finalresults)){
  cat(m ,"mean:",mean(finalresults[,m]),"std:",sd(finalresults[,m]), "mean+std:",mean(finalresults[,m])-sd(finalresults[,m]),"\n")
}