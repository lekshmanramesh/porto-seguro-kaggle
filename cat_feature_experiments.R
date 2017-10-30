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
## Smoothing Function applied while target encoding
mod_target_encode=function(train, test, feature, k){
  mean_train=mean(label)
  a=data.frame(cbind(freq=table(train[,feature,with=F]),
                     mean_cv=tapply(label, train[,feature,with=F], mean)))
  a$smoothing=(1/(1+exp((k-a$freq)/10)))
  a$final_mean_cv=a$smoothing*(a$mean_cv)+(1-a$smoothing)*mean_train
  eval(parse(text = paste0('write.csv(a,"',feature,'.csv")')))
  a_df=data.frame(name_char=row.names(a), val=a[,4])
#Train Data Feature Addition  
  feature_df=data.frame(feature=train[,feature,with=F])
  feature_df$id=1:nrow(train)
  check=merge(feature_df,a_df,by.x=feature,by.y="name_char",all.x = T)
  check=check[order(check$id),]
  train_feature=check$val
#Test Data Feature Addition  
  feature_df=data.frame(feature=test[,feature,with=F])
  feature_df$id=1:nrow(test)
  check=merge(feature_df,a_df,by.x=feature,by.y="name_char",all.x = T)
  check=check[order(check$id),]
  test_feature=check$val  
  return(list(train_feature,test_feature))
}
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
cv_smooth_target_encoding=NULL
select_features_exp=function(train, test, feature){
  if ((feature)!="None"){
    train$check0=unlist(mod_target_encode(train, test, feature, 50)[1])
    test$check0=unlist(mod_target_encode(train, test, feature, 50)[2])
    }

  train_data=as.matrix(train)
  test_data=as.matrix(test)
  xgmat=xgb.DMatrix(train_data,label=label, missing="NAN")
  xgmat_test=xgb.DMatrix(test_data,label=label, missing="NAN")  
  mod1=xgb.train(data=xgmat, label=label,nround=150, params = param, verbose=0)
  predicted=predict(mod1,xgmat_test)
  cv_inter=data.frame(name_feature=feature,metric=NormalizedGini(predicted,actual))
  return(cv_inter)
}

## For Loop to check each categorical variable

for (i in (1:nrow(class_cat))){
  cv_smooth_target_encoding=rbind(cv_smooth_target_encoding,
                                  select_features_exp(train_red, test_red, as.character(class_cat$features[i])))
}

## Without target encoding
cv_smooth_target_encoding=rbind(cv_smooth_target_encoding,
                                select_features_exp(train_red, test_red, "None"))
## All variables target encoded

for (i in (1:nrow(class_cat))){
  feature=as.character(class_cat$features[i])
  train_red[,feature, with=F]=unlist(mod_target_encode(train_red, test_red, feature, 50)[1])
  test_red[,feature, with=F]=unlist(mod_target_encode(train_red, test_red, feature, 50)[2])
}

cv_smooth_target_encoding=rbind(cv_smooth_target_encoding,
                                select_features_exp(train_red, test_red, "None"))