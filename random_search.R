library(ggplot2) # Data visualization
library(readr) # CSV file I/O, e.g. the read_csv function
library(data.table)
library(Matrix)
library(xgboost)
library(caret)
library(dplyr)
library(MLmetrics)

#Kevins Features :)

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
for (i in (1:nrow(class_cat))){
  feature=as.character(class_cat$features[i])
  eval(parse(text=paste0("data$",feature,"=as.character(data$",feature,")")))
}

param_data=data.frame(max_depth=sample(2:30, 20, replace = T),
                      subsample=runif(20),
                      colsample_bytree=runif(20))

train_data=sparse.model.matrix(target~.-1,data=data)
label=data$target
xgmat=xgb.DMatrix(train_data,label=label, missing="NAN")

xgb_normalizedgini <- function(preds, dtrain){
  actual <- getinfo(dtrain, "label")
  score <- NormalizedGini(preds,actual)
  return(list(metric = "NormalizedGini", value = score))
}

cv_final=NULL
View(cv_final)
for (i in (1:1)){
  param <- list(objective = "binary:logistic",  label=label,
                booster = "gbtree", eta = 0.1,
                subsample = param_data$subsample[i],
                colsample_bytree = param_data$colsample_bytree[i],
                max_depth = param_data$max_depth[i],
                min_child_weight = 0.77,
                alpha=8,
                lambda=1.3,
                scale_pos_weight=1.6)
  cv.res_red=xgb.cv(data=xgmat, nfold=2,nround=200, feval=xgb_normalizedgini, params = param, verbose=0)
  cv_op=cv.res_red$evaluation_log
  cv_op=cv_op[c(150,175,200),]
  cv_final=rbind(cv_final,cv_op)
}
