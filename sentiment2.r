setwd("/home/dan/Thesis")
source("helper.R")
source("PANDO.r")
source("PANDO2.r")
library(MASS)
library(grpreg)
library(xtable)
library(knitr)


# schools2=read.csv("sentimenttfidf.csv",stringsAsFactors = FALSE) # with sparsity factor 0.9
# 
# schools2[schools2[,"Label"] %in% c(1,2),"Label"]=-1
# schools2[schools2[,"Label"] %in% c(4,5),"Label"]=1
# schools2[,"Label"]=as.numeric(schools2[,"Label"])
# schools2=schools2[sample(nrow(schools2)),]


schools2=read.csv("sentiment_binary_tfidf.csv",stringsAsFactors = FALSE) # with sparsity factor 0.9


alltests=c()
NUM_TESTS=1
for(l in 1:NUM_TESTS){
  #set.seed(l+1)
  set.seed(l+11)
  testidx = c()
  validx = c()
  trainidx = c()
  for(fam in unique(schools2[,"Family"])){
    
    testidx = c(testidx, sample(which(schools2[,"Family"]==fam),floor(length(which(schools2[,"Family"]==fam))*0.75) ))
    famtrain = setdiff(which(schools2[,"Family"]==fam),testidx)
    famvalidx = sample(famtrain,0.25*length(famtrain) ) #val 10% of train
    validx = c(validx,famvalidx)
    famtrain = setdiff(famtrain,validx) # remove validation from train
    trainidx = c(trainidx,famtrain)
  }
  
  
  
  data=list()
  data[["data"]]=schools2
  data[["testidx"]]=testidx
  data[["trainidx"]]=trainidx
  data[["validx"]]=validx
  
  controls=rpart.control()
  iter=100
  rate=0.01
  ridge.lambda=0.0001  
  #data=GenerateData(d=d,ntrain=ntrain,ntest=ntest,seed=i)
  
  
  train = data$data[data$trainidx,]
  test = data$data[data$testidx,]
  
  val = data$data[data$validx,]
  cat("starting pando\n")
  mshared=TrainMultiTaskClassificationGradBoost(train,iter=iter,v=rate,groups=train[,"Family"],controls=controls,ridge.lambda=ridge.lambda,target="binary",valdata=val,fitTreeCoef = FALSE)
  bestIt=min(which(as.vector(mshared$log$vscore)==max(as.vector(mshared$log$vscore))))    
  mshared=TrainMultiTaskClassificationGradBoost(rbind(train,val),iter=bestIt,v=rate,groups=rbind(train,val)[,"Family"],controls=controls,ridge.lambda=ridge.lambda,target="binary",valdata=NULL,fitTreeCoef = FALSE)
  
  
  cat("starting pando2\n")
  mshared2=TrainMultiTaskClassificationGradBoost2(train,iter=iter,v=rate,groups=train[,"Family"],controls=controls,ridge.lambda=ridge.lambda,target="binary",treeType="rpart",valdata=val, fitCoef = "ridge")
  bestIt=min(which(as.vector(mshared2$log$vscore)==max(as.vector(mshared2$log$vscore))))    
  mshared2=TrainMultiTaskClassificationGradBoost2(rbind(train,val),iter=bestIt,v=rate,groups=rbind(train,val)[,"Family"],controls=controls,ridge.lambda=ridge.lambda,target="binary",treeType="rpart",valdata=NULL,fitCoef = "ridge")
  
  cat("starting pando3\n")
  #mshared3=TrainMultiTaskClassificationGradBoost(train,iter=iter,v=rate,groups=train[,"Family"],controls=rpart.control(maxdepth = 2, cp=0.01),ridge.lambda=ridge.lambda,target="binary",valdata=val,fitTreeCoef = TRUE)
  
  cat("starting pando4\n")
  #mshared4=TrainMultiTaskClassificationGradBoost(train,iter=iter,v=rate,groups=train[,"Family"],controls=rpart.control(maxdepth = 2, cp=0.01),ridge.lambda=2,target="binary",valdata=val,fitTreeCoef = FALSE)
  
  
  
  
  cat("starting per task models\n")
  #mshared3=TrainMultiTaskClassificationGradBoost2(train,iter=iter,v=rate,groups=train[,"Family"],controls=ctree_control(maxdepth = 4),ridge.lambda=ridge.lambda,target="regression",treeType="ctree",fitCoef="norm2")
  #mshared4=TrainMultiTaskClassificationGradBoost2(train,iter=iter,v=rate,groups=train[,"Family"],controls=ctree_control(maxdepth = 4),ridge.lambda=1,target="regression",treeType="ctree",fitCoef="norm1")
  
  perTaskModels=list()
  logitModels=list()
  for(fam in unique(train[,"Family"])){
    
    cat("fam ",fam,"\n")
    tr = train[train[,"Family"]==fam,]
    tr.val = val[val[,"Family"]==fam,]
    m0 = TrainMultiTaskClassificationGradBoost(tr,valdata=tr.val,groups = matrix(fam,nrow=nrow(tr),ncol=1),iter=iter,v=0.01,
                                               controls=controls, ridge.lambda = ridge.lambda,target="binary")  
    perTaskModels[[toString(fam)]]=m0
    logitModels[[toString(fam)]]= cv.glmnet(x=as.matrix(tr[,-which(colnames(tr) %in% c("Family","Label"))]),y=tr[,"Label"],family="binomial",alpha=1,maxit=10000,nfolds=3, thresh=1E-4)
  }
  
  ### train binary model, ignoring multi tasking:
  binaryData = train
  binaryData["Family"]="1"
  binaryVal = val
  binaryVal["Family"]="1"
  cat("starting binary boosting\n")
  
  mbinary=TrainMultiTaskClassificationGradBoost(binaryData,iter=iter,v=rate,groups=matrix(1,nrow=nrow(binaryData),ncol=1),controls=controls,ridge.lambda=ridge.lambda,target="binary",valdata=binaryVal)  
  bestIt=min(which(as.vector(mbinary$log$vscore)==max(as.vector(mbinary$log$vscore))))    
  mbinary=TrainMultiTaskClassificationGradBoost(rbind(binaryData,binaryVal),iter=bestIt,v=rate,groups=matrix(1,nrow=nrow(binaryData)+nrow(binaryVal),ncol=1),controls=controls,ridge.lambda=ridge.lambda,target="binary",valdata=NULL)  
  
  mlogitbinary = cv.glmnet(x=as.matrix(rbind(binaryData,binaryVal)[,-which(colnames(tr) %in% c("Family","Label"))]),y=rbind(binaryData,binaryVal)[,"Label"],family="binomial",alpha=1,maxit=10000,nfolds=3, thresh=1E-4)
  
  cat("create gplasso train data\n")
  gplassotraindata = CreateGroupLassoDesignMatrix(train)
  gplassoX = (gplassotraindata$X)[,-ncol(gplassotraindata$X)]
  gplassoy =  (gplassotraindata$X)[,ncol(gplassotraindata$X)]
  gplassoy[gplassoy==-1]=0
  
  mgplasso = cv.grpreg(gplassoX, gplassoy, group=gplassotraindata$groups, nfolds=2, maxit=10000,seed=777,family="binomial",trace=TRUE)
  rm(gplassotraindata) # remove gplasso train data, it can be very large
  gplassotestdata = CreateGroupLassoDesignMatrix(test)
  gplassotestX = (gplassotestdata$X)[,-ncol(gplassotestdata$X)]
  gplassotesty =  (gplassotestdata$X)[,ncol(gplassotestdata$X)]
  
  gplassoPreds = predict(mgplasso,gplassotestX,type="response",lambda=mgplasso$lambda.min)
  rm(gplassotestdata) # remove gplasso test data, it can be very large
  methods = c("PANDO","PTB","BB","PTLogit","BinaryLogit","PANDO2","GL")#"PANDO3","PANDO4")#"PANDO3","GL","PANDO4")
  
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
    compmatrix = matrix(NA,nrow=length(methods),ncol = length(methods))
    
    tr.test = test[test["Family"]==fam,]
    tr.test = tr.test[,-which(colnames(tr.test)=="Family")]
    
    
    bestIt=NULL
    tt[[methods[which(methods=="PANDO")]]]= predict(mshared[[toString(fam)]],tr.test[,-which(colnames(tr.test) %in% c("Family","Label"))],calibrate=TRUE,bestIt=bestIt)
    tt[[methods[which(methods=="PTB")]]]= predict(perTaskModels[[toString(fam)]][[toString(fam)]],tr.test[,-which(colnames(tr.test) %in% c("Family","Label"))],calibrate=TRUE)
    
    
    tt[[methods[which(methods=="BB")]]] = predict(mbinary[[toString(1)]],tr.test[,-which(colnames(tr.test) %in% c("Family","Label"))],calibrate=TRUE,bestIt=bestIt)
    tt[[methods[which(methods=="PTLogit")]]] =predict(logitModels[[toString(fam)]],newx=as.matrix(tr.test[,-which(colnames(tr) %in% c("Family","Label"))]),type="response",s=logitModels[[toString(fam)]]$lambda.min)
    tt[[methods[which(methods=="BinaryLogit")]]] =predict(mlogitbinary,newx=as.matrix(tr.test[,-which(colnames(tr) %in% c("Family","Label"))]),type="response",s=mlogitbinary$lambda.min)
    
    
    tt[[methods[which(methods=="PANDO2")]]] = predict(mshared2[[toString(fam)]],tr.test[,-which(colnames(tr.test) %in% c("Family","Label"))],calibrate=TRUE,bestIt=bestIt)
    #tt[[methods[7]]] = predict(mshared3[[toString(fam)]],tr.test[,-which(colnames(tr.test) %in% c("Family","Label"))],calibrate=TRUE)
    tt[[methods[which(methods=="GL")]]] = gplassoPreds[test[,"Family"]==fam]
    
    #bestIt=min(which(as.vector(mshared3$log$vscore)==max(as.vector(mshared3$log$vscore))))    
    #tt[[methods[which(methods=="PANDO3")]]] = predict(mshared3[[toString(fam)]],tr.test[,-which(colnames(tr.test) %in% c("Family","Label"))],calibrate=TRUE,bestIt=bestIt)
    
    #bestIt=min(which(as.vector(mshared4$log$vscore)==max(as.vector(mshared4$log$vscore))))    
    #tt[[methods[which(methods=="PANDO4")]]] = predict(mshared4[[toString(fam)]],tr.test[,-which(colnames(tr.test) %in% c("Family","Label"))],calibrate=TRUE,bestIt=bestIt)
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
        #cat("setting compmatrix",i," ",j,"\n")
        compmatrix[i,j] = signif(pROC::roc.test(rc[[methods[i]]],rc[[methods[j]]])$p.value, digits = 3)
        if(rc[[methods[i]]]$auc[1] < rc[[methods[j]]]$auc[1]){
          compmatrix[i,j] = compmatrix[i,j]*-1
        }
        cat("auc  for ",fam," ", methods[i]," VS ",methods[j],": ",round((rc[[methods[i]]]$auc[1]),4)," ",round((rc[[methods[j]]]$auc[1]),4)," with pval: ",pROC::roc.test(rc[[methods[i]]],rc[[methods[j]]])$p.value,"\n")
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
#colnames(finalAUC)=methods
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


compmat2=matrix(NA,nrow=length(unique(test[,"Family"])),ncol=length(methods))
rownames(compmat2)=unique(test[,"Family"])
colnames(compmat2)=methods
for(fam in unique(test[,"Family"])){
  testidxs = (test["Family"]==fam)
  for(method in methods){
    compmat2[which(rownames(compmat2)==fam),which(colnames(compmat2)==method)]=roc(as.factor(alltests[(alltests[,"testnum"]==l)&(testidxs),"Label"]),alltests[(alltests[,"testnum"]==l)&(testidxs),method])$auc[1]
    
  }
}


#### seed=1 or 2 got BB vs PANDO2 with 
# Z = -2.2629, p-value = 0.02364
# alternative hypothesis: true difference in AUC is not equal to 0
# sample estimates:
#   AUC of roc1 AUC of roc2 
# 0.9796171   0.9823319 

roc.test(roc(as.factor(alltests[alltests[,"testnum"]==l,"Label"]),alltests[alltests[,"testnum"]==l,"BB"]),
         roc(as.factor(alltests[alltests[,"testnum"]==l,"Label"]),alltests[alltests[,"testnum"]==l,"PANDO2"]))



#save.image(compress=TRUE)



for(m in methods){
  cat(m," ",mean(compmat2[,m]),"\n")
}




