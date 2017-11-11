library(data.table)
library(readr)
train <- read_csv("D:/Lekshman/Porto Seguro Insurance/train.csv", col_types = cols(id = col_skip()))
#Sampling to save time
train=train[sample(nrow(train),300000),]
train=data.table(train)
library(Matrix)
library(xgboost)
data=data.table(train)

data[, amount_nas := rowSums(data == -1, na.rm = T)]
data[, high_nas := ifelse(amount_nas>4,1,0)]
data[, ps_car_13_ps_reg_03 := ps_car_13*ps_reg_03]
data[, ps_reg_mult := ps_reg_01*ps_reg_02*ps_reg_03]
data[, ps_ind_bin_sum := ps_ind_06_bin+ps_ind_07_bin+ps_ind_08_bin+ps_ind_09_bin+
       ps_ind_10_bin+ps_ind_11_bin+ps_ind_12_bin+ps_ind_13_bin+ps_ind_16_bin+ps_ind_17_bin+ps_ind_18_bin]
data[,reg02_car13:=ps_reg_02+ps_car_13]
data[,reg01_car13:=ps_reg_01+ps_car_13]
data[,reg03_car15:=ps_reg_03+ps_car_15]
data[,reg02_car15:=ps_reg_02+ps_car_15]
data[,car12_car13:=ps_car_12*ps_car_13]
data[,car12_car15:=ps_car_12*ps_car_15]
data[,car12_car03:=ps_car_12*ps_car_13]
data[,car15_car03:=ps_car_15*ps_car_13]
data=as.data.frame(lapply(data, as.numeric))
#Converting character variables into character format
for (i in (1:nrow(class_cat))){
  feature=as.character(class_cat$features[i])
  eval(parse(text=paste0("data$",feature,"=as.character(data$",feature,")")))
}
#Best parameters from Kaggle Kernel
param <- list(booster="gbtree",
              objective="binary:logistic",
              eta = 0.1,
              gamma = 10,
              max_depth = 4,
              min_child_weight = 0.77,
              subsample = 0.8,
              colsample_bytree = 0.8,
              alpha=8,
              lambda=1.3,
              scale_pos_weight=1.6
)

#Normalized Gini Function
library(MLmetrics)
xgb_normalizedgini <- function(preds, dtrain){
  actual <- getinfo(dtrain, "label")
  score <- NormalizedGini(preds,actual)
  return(list(metric = "NormalizedGini", value = score))
}

#Function that drops 1 feature at a time and outputs XGB Performance
drop_features=function(feed_train,feature1){
  if (feature1 != "target") {eval(parse(text=paste0("feed_train$",feature1,"=NULL")))}
  train_data=sparse.model.matrix(target~.-1,data=feed_train)
  label=feed_train$target
  xgmat=xgb.DMatrix(train_data,label=label, missing="NAN")

  cv.res_red2=xgb.cv(data=xgmat, nfold=2, label=label,nround=150, 
                     params = param, verbose=0, feval=xgb_normalizedgini, 
                     early_stopping_rounds = 20, maximize = T)
  cv_op=cv.res_red2$evaluation_log
  cv_op=cv_op[nrow(cv_op),]
  cv_op$new_var=feature1
  return(cv_op)
}

cv_final_drop=NULL
for (i in (1:ncol(data))){
  cv_final_drop=rbind(cv_final_drop,drop_features(feed_train=data,colnames(data)[i]))
}