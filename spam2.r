setwd("/home/dan/Thesis")
source("helper.R")
source("PANDO.r")
source("PANDO2.r")
source("pando3.r")
library(MASS)
library(grpreg)
library(xtable)
library(knitr)


## http://www.jmlr.org/papers/volume8/xue07a/xue07a.pdf
## families 1-15 are from highliy foliated regions
## families 16-29 are from earth/desert regions
schools2=read.csv("spamtfidf.csv",stringsAsFactors = FALSE) # with sparsity factor 0.9
schools2=schools2[,-1]
schools2[schools2[,"Label"]==0,"Label"]=-1

alltests=c()
set.seed(777)
testidx = c()
validx = c()
trainidx = c()
l=1
for(fam in unique(schools2[,"Family"])){
  
  testidx = c(testidx, sample(which(schools2[,"Family"]==fam),floor(length(which(schools2[,"Family"]==fam))*0.9) ))
  famtrain = setdiff(which(schools2[,"Family"]==fam),testidx)
  famvalidx = sample(famtrain,0.15*length(famtrain) ) #val 10% of train
  validx = c(validx,famvalidx)
  famtrain = setdiff(famtrain,validx) # remove validation from train
  trainidx = c(trainidx,famtrain)
}

data=list()
data[["data"]]=schools2
data[["testidx"]]=testidx
data[["trainidx"]]=trainidx
data[["validx"]]=validx
iter=5000
rate=0.01
ridge.lambda=1  
train = data$data[data$trainidx,]
test = data$data[data$testidx,]
val = data$data[data$validx,]
cat("starting pando\n")
mshared=TunePando(TrainMultiTaskClassificationGradBoost,train,val,fitTreeCoef="nocoef",fitLeafCoef="ridge",trainrate = 0.01)
cat("starting pando2\n")
mshared2=TunePando(TrainMultiTaskClassificationGradBoost2,train,val,fitTreeCoef="ridge",fitLeafCoef="nocoef",trainrate = 0.01)
cat("starting per task models\n")

perTaskMethods=TRUE
if(perTaskMethods){
  perTaskModels=list()
  logitModels=list()
  for(fam in unique(train[,"Family"])){
    
    cat("fam ",fam,"\n")
    tr = train[train[,"Family"]==fam,]
    tr.val = val[val[,"Family"]==fam,]
    #    m0 = TrainMultiTaskClassificationGradBoost(tr,valdata=tr.val,groups = matrix(fam,nrow=nrow(tr),ncol=1),iter=iter,v=0.01,
    #                                               controls=rpart.control(), ridge.lambda = ridge.lambda,target="binary") 
    m0=TunePando(vanillaboost2,tr,tr.val,trainrate = 0.01)
    # m0 = TrainMultiTaskClassificationGradBoost(tr,valdata=tr.val,groups = matrix(fam,nrow=nrow(tr),ncol=1),iter=iter,v=0.01,
    #                                            controls=rpart.control(), ridge.lambda = ridge.lambda,target="binary")  
    
    perTaskModels[[toString(fam)]]=m0
    logitModels[[toString(fam)]]= cv.glmnet(x=as.matrix(rbind(tr,tr.val)[,-which(colnames(tr) %in% c("Family","Label"))]),y=rbind(tr,tr.val)[,"Label"],family="binomial",alpha=1,maxit=10000,nfolds=5, thresh=1E-6,nlambda = 50)
  }
  
}

### train binary model, ignoring multi tasking:
binaryData = train
binaryData["Family"]="1"
binaryVal = val
binaryVal["Family"]="1"
#mbinary=TrainMultiTaskClassificationGradBoost(binaryData,iter=iter,v=rate,groups=matrix(1,nrow=nrow(binaryData),ncol=1),controls=rpart.control(),ridge.lambda=ridge.lambda,target="binary",valdata=binaryVal)
mbinary=TunePando(vanillaboost2,binaryData,binaryVal,fitTreeCoef="nocoef",fitLeafCoef="nocoef",trainrate=0.01)
#mbinary2=TunePando(vanillaboost2,binaryData,binaryVal,fitTreeCoef="nocoef",fitLeafCoef="ridge",trainrate=0.1)

linearMethods=TRUE
if(linearMethods){
  mlogitbinary = cv.glmnet(x=as.matrix(rbind(binaryData,binaryVal)[,-which(colnames(tr) %in% c("Family","Label"))]),y=rbind(binaryData,binaryVal)[,"Label"],family="binomial",alpha=1,maxit=100000,nfolds=5, nlambda=100,thresh=1E-6)
  gplassotraindata = CreateGroupLassoDesignMatrix(rbind(train,val))
  gplassoX = (gplassotraindata$X)[,-ncol(gplassotraindata$X)]
  gplassoy =  (gplassotraindata$X)[,ncol(gplassotraindata$X)]
  gplassoy[gplassoy==-1]=0
  
  mgplasso = cv.grpreg(gplassoX, gplassoy, group=gplassotraindata$groups, nfolds=3, maxit=10000,seed=777,family="binomial",trace=TRUE,penalty="grLasso")
  
  gplassotestdata = CreateGroupLassoDesignMatrix(test)
  gplassotestX = (gplassotestdata$X)[,-ncol(gplassotestdata$X)]
  gplassotesty =  (gplassotestdata$X)[,ncol(gplassotestdata$X)]
  
  gplassoPreds = predict(mgplasso,gplassotestX,type="response",lambda=mgplasso$lambda.min)
  
  
  
}
m1 = if(linearMethods) c("BinaryLogit","GL") else c()
m2 = if(perTaskMethods) c("PTB","PTLogit") else c()
methods = c(c("PANDO","PANDO2","BB"),m1,m2)

rc=list()
tt=list()
compmat = c()
digitsfmt = matrix(-2,nrow=length(methods),ncol=length(methods))
xtables=list()
k=0
allpreds = matrix(nrow=nrow(test),ncol=length(methods)+2)
#colnames(allpreds)=c(methods,"Label","testnum")
colnames(allpreds)=c(methods,"Label","Family")
allpreds[,"Label"]=test[,"Label"]
allpred[,"Family"]=test[,"Family"]
#allpreds[,"testnum"]=l



##################### test:
for(fam in unique(test[,"Family"])){
  k = k+1
  testidxs = which(test["Family"]==fam)
  compmatrix = matrix(NA,nrow=length(methods),ncol = length(methods))
  
  tr.test = test[test["Family"]==fam,]
  tr.test = tr.test[,-which(colnames(tr.test)=="Family")]

  if("PTB" %in% methods){
    tt[[methods[which(methods=="PTB")]]]= predict(perTaskModels[[toString(fam)]][[toString(fam)]],tr.test[,-which(colnames(tr.test) %in% c("Family","Label"))],calibrate=TRUE)#,bestIt=bestIt)
  }
  if("BB" %in% methods){
    tt[[methods[which(methods=="BB")]]] = predict(mbinary[[toString(1)]],tr.test[,-which(colnames(tr.test) %in% c("Family","Label"))],calibrate=TRUE)#,bestIt=bestIt)
  }
  if("BB2" %in% methods){
    tt[[methods[which(methods=="BB2")]]] = predict(mbinary2[[toString(1)]],tr.test[,-which(colnames(tr.test) %in% c("Family","Label"))],calibrate=TRUE)#,bestIt=bestIt)
  }  
  if("PTLogit" %in% methods){
    tt[[methods[which(methods=="PTLogit")]]] = predict(logitModels[[toString(fam)]],newx=as.matrix(tr.test[,-which(colnames(tr) %in% c("Family","Label"))]),type="response",s=logitModels[[toString(fam)]]$lambda.min)
  }
  if("BinaryLogit" %in% methods){
    tt[[methods[which(methods=="BinaryLogit")]]] = predict(mlogitbinary,newx=as.matrix(tr.test[,-which(colnames(tr) %in% c("Family","Label"))]),type="response",s=mlogitbinary$lambda.min)
  }
  if("PANDO" %in% methods){
    tt[[methods[which(methods=="PANDO")]]] = predict(mshared[[toString(fam)]],tr.test[,-which(colnames(tr.test) %in% c("Family","Label"))],calibrate=TRUE)#,bestIt=bestIt)
  }
  if("PANDO2" %in% methods){
    tt[[methods[which(methods=="PANDO2")]]] = predict(mshared2[[toString(fam)]],tr.test[,-which(colnames(tr.test) %in% c("Family","Label"))],calibrate=TRUE)#,bestIt=bestIt)
  }
    
  if("GL" %in% methods){
    tt[[methods[which(methods=="GL")]]] = gplassoPreds[test[,"Family"]==fam]
  }
  
  


  
  

  
  
  
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
      cat("auc  for ",fam," ", methods[i]," VS ",methods[j],": ",round((rc[[methods[i]]]$auc[1]),4)," ",round((rc[[methods[j]]]$auc[1]),4),"\n")
    }
  }
  #compmat = rbind(compmat,compmatrix)
  compmat[[fam]] = compmatrix
  cat("***********\n")
  

  dft=data.frame(compmatrix)
  colnames(dft)=methods
  rownames(dft)=methods
  xtables[[toString(k)]]=xtable(dft,digits=cbind(rep(1,nrow(digitsfmt)),digitsfmt))

}
#####
allpreds[,"Family"]=test[,"Family"]
alltests=rbind(alltests,allpreds)
# cat("round ",l," summary:\n********************\n")
# for(method in methods){
#   score=pROC::roc(as.factor(alltests[alltests[,"testnum"]==l,"Label"]),alltests[alltests[,"testnum"]==l,method])$auc[1]
#   cat(method," ",score,"\n")
# }
families=unique(test[,"Family"])
for(method in methods){
  aucs=0
  for(fam in families){
    idx=which(method==methods)
    aucs = aucs+(compmat[[fam]])[idx,idx]
  }
  avgauc=aucs/length(families)
  cat("avg auc for ",method,":",avgauc,"\n")
}
  

### for bootstrapping: y_pred --> prediction, y_test--> label, groups --> the family of each instance. usually groups = test[,Family]
avgauc = function(allpreds){
  families=unique(allpreds[,"Family"])
  ret=c()
  for(method in methods){
    aucavg=0
    for(fam in families){
      idxs=allpreds[,"Family"]==fam
      aucscore=pROC::roc(as.factor(allpreds[idxs,"Label"]),as.numeric(allpreds[idxs,method]))$auc[1]  
      aucavg = aucavg+aucscore
    }
    aucavg=aucavg/length(families)
    ret = c(ret,aucavg)
  }
  ret=matrix(ret,nrow=1)
  colnames(ret)=methods
  return(ret)
}



allavgs=c()
allidxs=c()
for(i in 1:50){
  for(fam in families){
    idxs=which((allpreds[,"Family"]==fam))
    selectedidxs=sample(idxs,replace=TRUE,size=length(idxs))
    allidxs = c(allidxs,selectedidxs)
  }
  allavgs=rbind(allavgs,avgauc(allpreds[allidxs,]))
}

quantile(allavgs[,"PANDO2"]-allavgs[,"BB"],probs=c(0.025,0.975)) ### this gives us the 95% confidence interval
quantile(allavgs[,"PANDO2"]-allavgs[,"BinaryLogit"],probs=c(0.025,0.975)) ### this gives us the 95% confidence interval
  

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
#   AUC of roc1   AUC of roc2 
# 0.9796171   0.9823319 
allpreds=data.frame(allpreds)
roc.test(roc(as.factor(allpreds[,"Label"]),as.numeric(allpreds[,"BB"])),
         roc(as.factor(allpreds[,"Label"]),as.numeric(allpreds[,"PANDO2"])))


roc.test(roc(as.factor(allpreds[,"Label"]),as.numeric(allpreds[,"BinaryLogit"])),
         roc(as.factor(allpreds[,"Label"]),as.numeric(allpreds[,"PANDO2"])))



roc.test(roc(as.factor(alltests[alltests[,"testnum"]==l,"Label"]),alltests[alltests[,"testnum"]==l,"BB"]),
         roc(as.factor(alltests[alltests[,"testnum"]==l,"Label"]),alltests[alltests[,"testnum"]==l,"PANDO2"]))


roc.test(roc(as.factor(alltests[alltests[,"testnum"]==l,"Label"]),alltests[alltests[,"testnum"]==l,"GL"]),
         roc(as.factor(alltests[alltests[,"testnum"]==l,"Label"]),alltests[alltests[,"testnum"]==l,"BinaryLogit"]))


roc.test(roc(as.factor(alltests[alltests[,"testnum"]==l,"Label"]),alltests[alltests[,"testnum"]==l,"BinaryLogit"]),
         roc(as.factor(alltests[alltests[,"testnum"]==l,"Label"]),alltests[alltests[,"testnum"]==l,"PTLogit"]))


roc.test(roc(as.factor(alltests[alltests[,"testnum"]==l,"Label"]),alltests[alltests[,"testnum"]==l,"GL"]),
         roc(as.factor(alltests[alltests[,"testnum"]==l,"Label"]),alltests[alltests[,"testnum"]==l,"PTLogit"]))

dff = FeatureImportance.BoostingModel(mshared2_obo$fam1) 
PlotImp(dff,flip = TRUE)
dev.copy2eps(file="plots/SpamFeatureImportancePANDO.eps")

dff = PerTaskImportances(perTaskModels) 
PlotImp(dff,flip = TRUE)
dev.copy2eps(file="plots/SpamFeatureImportancePTB.eps")

#save.image(compress=TRUE)

for(m in methods){
  cat(m," ",mean(compmat2[,m]),"\n")
}



