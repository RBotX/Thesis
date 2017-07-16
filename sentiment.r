library(Matrix)
library(tm)
#library(text2vec)
#library(lsa)
#library(tidytext)

df=read.csv("/home/dan/Thesis/sentiment/sentiment_binary.csv")
dff=df[,which(colnames(df) %in% c("row","col","val","Label","Family"))]
dff = transform(dff,
                row = factor(row),
                col = factor(col))

data.sparse = sparseMatrix(as.integer(dff$row), as.integer(dff$col), x = dff$val)
#data.sparse2 = 
colnames(data.sparse) = levels(dff$col)
rownames(data.sparse) = levels(dff$row)
dtm  = as.DocumentTermMatrix(data.sparse,weighting = weightTfIdf)
dtm2=removeSparseTerms(dtm,0.97) #### used to be 0.9
alldata=data.frame(as.matrix(dtm2))
alldata[,"Family"]=NA
#alldata[1,]=dff[dff[row==1]]
## add family to each data
## add Family to each row
alldata[,"Family"]=sapply(1:nrow(alldata), function(i) toString(unique(dff[(dff[,"row"]==i),"Family"])))
alldata[,"Label"]=sapply(1:nrow(alldata), function(i) toString(unique(dff[(dff[,"row"]==i),"Label"])))

write.csv(alldata,"/home/dan/Thesis/sentiment_binary_tfidf.csv")