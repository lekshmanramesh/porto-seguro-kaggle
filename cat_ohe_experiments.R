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

train_red=data[sample(nrow(train),200000),]
test_red=train_red[1:100000,]
train_red=train_red[100001:200000,]
label=train_red$target
actual=test_red$target
train_red$target=NULL
test_red$target=NULL

param <- list(booster="gbtree",
              objective="binary:logistic",
              eta = 0.08,
              gamma = 10,
              max_depth = 4,
              min_child_weight = 0.77,
              subsample = 0.8,
              colsample_bytree = 0.8,
              alpha=8,
              lambda=1.3,
              scale_pos_weight=1.6
)
cv_ohe=NULL
select_features_exp=function(train, test, feature){
  if ((feature)!="None"){
    train$check0=as.character(train[,feature,with=F])
    test$check0=as.character(test[,feature,with=F])
  }
  train_data=sparse.model.matrix(~.-1,data=train_red)
  test_data=sparse.model.matrix(~.-1,data=test_red)

  xgmat=xgb.DMatrix(train_data,label=label, missing="NAN")
  xgmat_test=xgb.DMatrix(test_data,label=label, missing="NAN")
  mod1=xgb.train(data=xgmat, label=label,nround=150, params = param, verbose=0)
  predicted=predict(mod1,xgmat_test)
  cv_inter=data.frame(name_feature=feature,metric=NormalizedGini(predicted,actual))
  return(cv_inter)
}

## For Loop to check each categorical variable
for (i in (1:nrow(class_cat))){
  cv_ohe=rbind(cv_ohe,select_features_exp(train_red, test_red, as.character(class_cat$features[i])))
}
## Without any encoding
cv_ohe=rbind(cv_ohe,select_features_exp(train_red, test_red, "None"))
