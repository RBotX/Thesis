setwd("/home/dan/Thesis")

library(data.table)
source("helper.R")
source("PANDO.r")
source("PANDO2.r")
library(MASS)
library(grpreg)
library(xtable)
library(knitr)
library(fastICA)
library(caret)
library(dummies)

generateData=FALSE
set.seed(1)
if(generateData){
  schools=fread("/home/dan/Thesis/merck.csv")
  schools=data.frame(schools)
  
  schools2=schools
  rm(schools)
  schools2 = schools2[sample(nrow(schools2)),]
  schools3 = schools2[sample(1:nrow(schools2),nrow(schools2)*0.3), ]
  nz=nearZeroVar(schools3)
  
  schools2=schools2[,-nz]
  write.csv(schools2,"/home/dan/Thesis/merck_nzv.csv")
}

schools2=read.csv("/home/dan/Thesis/merck_nzv.csv")
schools2[is.na(schools2)]=0
s=svd(na.omit(schools2))

tt = dummy.data.frame(schools2)

alltests=c()
NUM_TESTS=1
for(l in 1:NUM_TESTS){
  cat("train-test split number ",l,"\n")
  set.seed(l)
  testidx = c()
  for(fam in unique(schools2[,"Family"])){
    #http://aima.eecs.berkeley.edu/~russell/classes/cs294/f05/papers/evgeniou+al-2005.pdf they take 75-25 train-test split, cited 669 times
    testidx = c(testidx, sample(which(schools2[,"Family"]==fam),floor(length(which(schools2[,"Family"]==fam))*0.98) )) 
  }
  
  
  
  data=list()
  data[["data"]]=schools2
  data[["testidx"]]=testidx
  
  
  iter=10
  rate=0.01
  ridge.lambda=1  
  #data=GenerateData(d=d,ntrain=ntrain,ntest=ntest,seed=i)
  
  train = data$data[-data$testidx,]
  test = data$data[data$testidx,]
  
  mshared=TrainMultiTaskClassificationGradBoost(train,iter=iter,v=rate,groups=train[,"Family"],controls=rpart.control(maxdepth = 5),ridge.lambda=ridge.lambda,target="regression")
  mshared2=TrainMultiTaskClassificationGradBoost2(train,iter=iter,v=rate,groups=train[,"Family"],controls=ctree_control(maxdepth = 5),ridge.lambda=ridge.lambda,target="regression",treeType="ctree")
  mshared3=TrainMultiTaskClassificationGradBoost2(train,iter=iter,v=rate,groups=train[,"Family"],controls=rpart.control(maxdepth = 5),ridge.lambda=ridge.lambda,target="regression",fitCoef="norm2")
  
  perTaskModels=list()
  logitModels=list()
  for(fam in unique(train[,"Family"])){
    
    cat("fam ",fam,"\n")
    tr = train[train[,"Family"]==fam,]
    m0 = TrainMultiTaskClassificationGradBoost(tr,groups = matrix(fam,nrow=nrow(tr),ncol=1),iter=100,v=rate,
                                               controls=rpart.control(maxdepth = 3), ridge.lambda = ridge.lambda,target="regression")  
    perTaskModels[[toString(fam)]]=m0
    logitModels[[toString(fam)]]= cv.glmnet(x=as.matrix(tr[,-which(colnames(tr) %in% c("Family","Label"))]),y=tr[,"Label"],family="gaussian",alpha=1,maxit=10000,nfolds=4, thresh=1E-4)
  }
  
  ### train binary model, ignoring multi tasking:
  binaryData = train
  binaryData["Family"]=1
  mbinary=TrainMultiTaskClassificationGradBoost(binaryData,iter=iter,v=rate,groups=matrix(1,nrow=nrow(binaryData),ncol=1),controls=rpart.control(maxdepth = 3),ridge.lambda=ridge.lambda,target="regression")  
  mlogitbinary = cv.glmnet(x=as.matrix(binaryData[,-which(colnames(tr) %in% c("Family","Label"))]),y=binaryData[,"Label"],family="gaussian",alpha=1,maxit=10000,nfolds=4, thresh=1E-4)
  
  gplassotraindata = CreateGroupLassoDesignMatrix(train)
  gplassoX = (gplassotraindata$X)[,-ncol(gplassotraindata$X)]
  gplassoy =  (gplassotraindata$X)[,ncol(gplassotraindata$X)]
  #gplassoy[gplassoy==-1]=0
  
  mgplasso = cv.grpreg(gplassoX, gplassoy, group=gplassotraindata$groups, nfolds=2, maxit=10000,seed=777,family="gaussian",trace=TRUE)
  
  gplassotestdata = CreateGroupLassoDesignMatrix(test)
  gplassotestX = (gplassotestdata$X)[,-ncol(gplassotestdata$X)]
  gplassotesty =  (gplassotestdata$X)[,ncol(gplassotestdata$X)]
  
  gplassoPreds = predict(mgplasso,gplassotestX,type="response",lambda=mgplasso$lambda.min)
  
  methods = c("PANDO","PTB","BB","PTLogit","BinaryLogit","PANDO2","PANDO3","GL")
  
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
    
    tt[[methods[1]]]= predict(mshared[[toString(fam)]],tr.test[,-which(colnames(tr.test) %in% c("Family","Label"))],calibrate=FALSE)
    tt[[methods[2]]]= predict(perTaskModels[[toString(fam)]][[toString(fam)]],tr.test[,-which(colnames(tr.test) %in% c("Family","Label"))],calibrate=FALSE)
    tt[[methods[3]]] = predict(mbinary[[toString(1)]],tr.test[,-which(colnames(tr.test) %in% c("Family","Label"))],calibrate=FALSE)
    tt[[methods[4]]] =predict(logitModels[[toString(fam)]],newx=as.matrix(tr.test[,-which(colnames(tr) %in% c("Family","Label"))]),type="response",s=logitModels[[toString(fam)]]$lambda.min)
    tt[[methods[5]]] =predict(mlogitbinary,newx=as.matrix(tr.test[,-which(colnames(tr) %in% c("Family","Label"))]),type="response",s=mlogitbinary$lambda.min)
    tt[[methods[6]]] = predict(mshared2[[toString(fam)]],tr.test[,-which(colnames(tr.test) %in% c("Family","Label"))],calibrate=FALSE)
    tt[[methods[7]]] = predict(mshared3[[toString(fam)]],tr.test[,-which(colnames(tr.test) %in% c("Family","Label"))],calibrate=FALSE)
    tt[[methods[8]]] = gplassoPreds[test[,"Family"]==fam]
    
    for(i in 1:length(methods)){
      rc[[methods[i]]]=sqrt(mean((tr.test[,"Label"]-tt[[methods[i]]])**2))
      allpreds[testidxs,methods[i]] = as.matrix(tt[[methods[i]]],ncol=1)
    }
    
    
    
    for(i in 1:length(methods)){
      #cat("storing ",round(pROC::auc(rc[[methods[i]]])[1],4)," in ",i,i,"\n" )
      compmatrix[i,i]=rc[[methods[i]]] 
      digitsfmt[i,i]=3
      for(j in 1:length(methods)){
        if(i >=j ){
          next
        }
        #cat("storing ",signif(pROC::roc.test(rc[[methods[i]]],rc[[methods[j]]])$p.value, digits = 3)," now in ",i,j,"\n")
        compmatrix[i,j] = signif((rc[[methods[i]]]-rc[[methods[j]]]), digits = 3)
        cat("RMSE for ",fam," ", methods[i]," VS ",methods[j],": ",round((rc[[methods[i]]])[1],4)," ",round((rc[[methods[j]]])[1],4),"\n")
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
  
  for(method in methods){
    score=mean(sqrt((tr.test[,"Label"]-tt[[method]])**2))
    cat(method," ",score,"\n")
  }
  
  alltests=rbind(alltests,allpreds)
  
}
cat("*********************************************\n")
cat("final results - RMSE:\n")
finalresults= matrix(nrow=NUM_TESTS,ncol=length(methods))
finalresults2= matrix(nrow=NUM_TESTS,ncol=length(methods))
colnames(finalresults)=methods
colnames(finalresults2)=methods
for(l in 1:NUM_TESTS){
  for(method in methods){
    
    score=mean(sqrt((alltests[alltests[,"testnum"]==l,"Label"]-alltests[alltests[,"testnum"]==l,method])**2))
    cat(method," ",score,"\n")
    finalresults[l,method]=score
  }
}
for(m in colnames(finalresults)){
  cat(m ,"mean:",mean(finalresults[,m]),"std:",sd(finalresults[,m]), "mean+std:",mean(finalresults[,m])+sd(finalresults[,m]),"\n")
}

###exaplined vairance
cat("final results - Explained Variance:\n")
for(l in 1:NUM_TESTS){
  for(method in methods){
    totalvar=var(schools2[,"Label"])
    score=(totalvar - var(alltests[alltests[,"testnum"]==l,"Label"]-alltests[alltests[,"testnum"]==l,method]))/(totalvar)
    
    finalresults2[l,method]=score
  }
}

for(m in colnames(finalresults2)){
  cat(m ,"mean:",mean(finalresults2[,m]),"std:",sd(finalresults2[,m]), "mean-std:",mean(finalresults2[,m])-sd(finalresults2[,m]),"\n")
}









