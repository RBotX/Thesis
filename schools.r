source("helper.r")
source("PANDO.r")
source("PANDO2.r")
library(MASS)
library(grpreg)
library(xtable)
library(knitr)


CreateGroupLassoDesignMatrix = function(X){
  
  families = unique(X[,"Family"])
  ntasks=length(families)
  colsPerTask = (ncol(X)-2) # removing label and family columns
  colsPerTask = colsPerTask+1 # adding intercept column per each task (two different rows of code just for readibility)
  Xgl = matrix(0,nrow=nrow(X),ncol=(colsPerTask*ntasks)+1) # the +1 is for the label
  #cat(nrow(Xgl),"XGL",ncol(Xgl),"\n")
  i=0
  
  rowend=0
  #cat("ncol of xgl is:",ncol(Xgl),"\n")
  groups=rep(1:colsPerTask,ntasks) # which group each variable belongs to. in our case, we group the same variable across tasks
  for(fam in families){
    #cat("fam is :" ,fam,"\n")
    nr = nrow(X[X[,"Family"]==fam,]) ## how many instances for this family/task
    nc = colsPerTask ## we know in our formulation that all tasks share the same number of features
    rowstart=(rowend+1)#(i*nr)+1
    rowend = rowstart + nr -1
    colstart = (i*nc)+1
    colend = (i+1)*nc
    #cat(rowstart,"->",rowend,"\n")
    #cat(colstart,"->",colend,"\n")
    Xgl[rowstart:rowend,colstart:(colend-1)]=as.matrix(X[X[,"Family"]==fam,-which(colnames(X) %in% c("Family","Label"))])
    Xgl[rowstart:rowend,colend]=matrix(1,nrow=length(rowstart:rowend),ncol=1) ## add intercept per task
    ## add label
    Xgl[rowstart:rowend,ncol(Xgl)] = as.matrix(X[X[,"Family"]==fam,"Label"])
    i=i+1
    
  }
  ret=list()
  ret[["X"]]=Xgl
  ret[["groups"]]=groups
  return(ret)
}

schools=read.csv("schools.csv")
schools=schools[,-1]


schools2=schools
colnames(schools2)[which(colnames(schools2)=="target")]="Label"
# for(task in unique(schools[,"task"])){
#   scores=schools2[schools2[,"task"]==task,"target"]
#   medianscore = median(scores)
#   schools2[schools2[,"task"]==task,"Label"]=(schools2[schools2[,"task"]==task,"target"]>medianscore)+0
#   schools2[schools2[,"target"]==0,"target"]=-1
# }


colnames(schools2)[which(colnames(schools2)=="task")]="Family"
schools2[,"Family"] = apply(as.matrix(schools2[,"Family"]),1,function(x){paste0("fam",toString(x))})


#schools2=schools2[,-which(colnames(schools2)=="target")]
fams = table(schools2[,"Family"])[table(schools2[,"Family"]) > 120]
fams = names(fams)
schools2=schools2[schools2[,"Family"] %in% fams,]
# s1=schools2[,-which(colnames(schools2) %in% c("Label","Family"))]
# s1 = cbind(s1,schools2[,"Family"])
# colnames(s1)[ncol(s1)]=paste0("X",ncol(s1)-1)
# schools2 = cbind(s1,schools2[,which(colnames(schools2) %in% c("Label","Family"))])
alltests=c()
NUM_TESTS=10
for(l in 1:NUM_TESTS){
  set.seed(l)
  testidx = c()
  for(fam in unique(schools2[,"Family"])){
    testidx = c(testidx, sample(which(schools2[,"Family"]==fam),floor(length(which(schools2[,"Family"]==fam))*0.4) ))
  }
  
  
  
  data=list()
  data[["data"]]=schools2
  data[["testidx"]]=testidx
  
  controls=rpart.control(maxdepth = 2)
  iter=300
  rate=0.01
  ridge.lambda=1  
  #data=GenerateData(d=d,ntrain=ntrain,ntest=ntest,seed=i)
  
  train = data$data[-data$testidx,]
  test = data$data[data$testidx,]
  
  mshared=TrainMultiTaskClassificationGradBoost(train,iter=iter,v=rate,groups=train[,"Family"],controls=controls,ridge.lambda=ridge.lambda,target="regression")
  mshared2=TrainMultiTaskClassificationGradBoost2(train,iter=iter,v=rate,groups=train[,"Family"],controls=controls,ridge.lambda=ridge.lambda,target="regression")
  
  perTaskModels=list()
  logitModels=list()
  for(fam in unique(train[,"Family"])){
    
    cat("fam ",fam,"\n")
    tr = train[train[,"Family"]==fam,]
    m0 = TrainMultiTaskClassificationGradBoost(tr,groups = matrix(fam,nrow=nrow(tr),ncol=1),iter=iter,v=rate,
                                               controls=controls, ridge.lambda = ridge.lambda,target="regression")  
    perTaskModels[[toString(fam)]]=m0
    logitModels[[toString(fam)]]= cv.glmnet(x=as.matrix(tr[,-which(colnames(tr) %in% c("Family","Label"))]),y=tr[,"Label"],family="gaussian",alpha=1,maxit=10000,nfolds=4, thresh=1E-4)
  }
  
  ### train binary model, ignoring multi tasking:
  binaryData = train
  binaryData["Family"]=1
  mbinary=TrainMultiTaskClassificationGradBoost(binaryData,iter=iter,v=rate,groups=matrix(1,nrow=nrow(binaryData),ncol=1),controls=controls,ridge.lambda=ridge.lambda,target="regression")  
  mlogitbinary = cv.glmnet(x=as.matrix(binaryData[,-which(colnames(tr) %in% c("Family","Label"))]),y=binaryData[,"Label"],family="gaussian",alpha=1,maxit=10000,nfolds=4, thresh=1E-4)
  
  gplassotraindata = CreateGroupLassoDesignMatrix(train)
  gplassoX = (gplassotraindata$X)[,-ncol(gplassotraindata$X)]
  gplassoy =  (gplassotraindata$X)[,ncol(gplassotraindata$X)]
  #gplassoy[gplassoy==-1]=0
  
  mgplasso = cv.grpreg(gplassoX, gplassoy, group=gplassotraindata$groups, nfolds=4, maxit=10000,seed=777,family="gaussian",trace=TRUE)
  
  gplassotestdata = CreateGroupLassoDesignMatrix(test)
  gplassotestX = (gplassotestdata$X)[,-ncol(gplassotestdata$X)]
  gplassotesty =  (gplassotestdata$X)[,ncol(gplassotestdata$X)]
  
  gplassoPreds = predict(mgplasso,gplassotestX,type="response",lambda=mgplasso$lambda.min)   
  methods = c("PANDO","PTB","BB","PTLogit","BinaryLogit","GL","PANDO2")
  
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
    #tr.test = test[test[,"Family"] %in% c(fam,"clean"),-which(colnames(test)=="Family")]
    tr.test = test[test["Family"]==fam,]
    tr.test = tr.test[,-which(colnames(tr.test)=="Family")]
    
    tt[[methods[1]]]= predict(mshared[[toString(fam)]],tr.test[,-which(colnames(tr.test) %in% c("Family","Label"))],calibrate=FALSE)
    allpreds[testidxs,methods[1]] = tt[[methods[1]]]
    rc[[methods[1]]] = sqrt(mean((tr.test[,"Label"]-tt[[methods[1]]])**2))
    
    
    tt[[methods[2]]]= predict(perTaskModels[[toString(fam)]][[toString(fam)]],tr.test[,-which(colnames(tr.test) %in% c("Family","Label"))],calibrate=FALSE)
    allpreds[testidxs,methods[2]] = tt[[methods[2]]]
    
    rc[[methods[2]]] = sqrt(mean((tr.test[,"Label"]-tt[[methods[2]]])**2))
    
    
    tt[[methods[3]]] = predict(mbinary[[toString(1)]],tr.test[,-which(colnames(tr.test) %in% c("Family","Label"))],calibrate=FALSE)
    allpreds[testidxs,methods[3]] = tt[[methods[3]]]
    
    rc[[methods[3]]] = sqrt(mean((tr.test[,"Label"]-tt[[methods[3]]])**2))
    tt[[methods[4]]] =predict(logitModels[[toString(fam)]],newx=as.matrix(tr.test[,-which(colnames(tr) %in% c("Family","Label"))]),type="response",s=logitModels[[toString(fam)]]$lambda.min)
    allpreds[testidxs,methods[4]] = tt[[methods[4]]]
    #rc[[methods[4]]] = pROC::roc(as.factor(tr.test[,"Label"]),as.numeric(tt[[methods[4]]]))
    rc[[methods[4]]] = sqrt(mean((tr.test[,"Label"]-tt[[methods[4]]])**2))
    tt[[methods[5]]] =predict(mlogitbinary,newx=as.matrix(tr.test[,-which(colnames(tr) %in% c("Family","Label"))]),type="response",s=mlogitbinary$lambda.min)
    allpreds[testidxs,methods[5]] = tt[[methods[5]]]
    #rc[[methods[5]]] = pROC::roc(as.factor(tr.test[,"Label"]),as.numeric(tt[[methods[5]]]))
    rc[[methods[5]]] = sqrt(mean((tr.test[,"Label"]-tt[[methods[5]]])**2))
    tt[[methods[6]]] = gplassoPreds[test[,"Family"]==fam]
    allpreds[testidxs,methods[6]] = tt[[methods[6]]]
    #rc[[methods[6]]] = pROC::roc(as.factor(tr.test[,"Label"]),as.numeric(tt[[methods[6]]]))
    rc[[methods[6]]] = sqrt(mean((tr.test[,"Label"]-tt[[methods[6]]])**2))
    
    tt[[methods[7]]] = predict(mshared2[[toString(fam)]],tr.test[,-which(colnames(tr.test) %in% c("Family","Label"))],calibrate=FALSE)
    allpreds[testidxs,methods[7]] = tt[[methods[7]]]
    rc[[methods[7]]] = sqrt(mean((tr.test[,"Label"]-tt[[methods[7]]])**2))
    
    
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
cat("final results:\n")
finalresults= matrix(nrow=NUM_TESTS,ncol=length(methods))
colnames(finalresults)=methods
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





