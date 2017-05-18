library(Matrix)
library(tm)
library(text2vec)
library(lsa)
library(tidytext)

df=read.csv("/home/dan/Thesis/spam/spam.csv")
dff=df[,which(colnames(df) %in% c("row","col","val","Family"))]
dff = transform(dff,
                                  row = factor(row),
                                  col = factor(col))

data.sparse = sparseMatrix(as.integer(dff$row), as.integer(dff$col), x = dff$val)
colnames(data.sparse) = levels(dff$col)
rownames(data.sparse) = levels(dff$row)
dtm  = as.DocumentTermMatrix(data.sparse,weight="weightTfIdf")

