# Function To Calculate IV
iv<-function(predit,target) # my somer's D function
{
  data<-data.frame(predit,target);
  data_sort<-data[order(predit),]
  
  ttl_num<-length(target);
  bin<-10;
  n<-ttl_num%/%bin;
  iv_bin<-rep(0,times=bin);
  good<-rep(0,times=bin);
  bad<-rep(0,times=bin);
  for (i in 1:bin) # calculate PSI for ith bin
  {
    if(i!=bin) {good[i]<-sum(data_sort$target[((i-1)*n+1):(n*i)]);bad[i]<-n-good[i]} else
    {good[i]<-sum(data_sort$target[((i-1)*n+1):ttl_num]);bad[i]<-ttl_num-n*(i-1)-good[i]}
  }
  
  good_pct<-good/sum(good)
  bad_pct<-bad/sum(bad)
  for (i in 1:bin)
  {
    iv_bin[i]<-(bad_pct[i]-good_pct[i])*log(bad_pct[i]/good_pct[i])
  }
  
  iv=sum(iv_bin)
  return (iv)
}

# Exploring combinations of numeric features
select_features_exp=function(data, feature1, feature2, sgn){
  eval(parse(text=paste0("data$",feature1,"_",feature2,"=",
                                                         paste0("data$",feature1,sgn,"data$",feature2))))
  feature1_vec=eval(parse(text=paste0("data$",feature1)))
  feature2_vec=eval(parse(text=paste0("data$",feature2)))
  IV=iv(eval(parse(text=paste0("data$",feature1,"_",feature2))),data$target)
  iv_feature1=iv(feature1_vec,data$target)
  iv_feature2=iv(feature2_vec,data$target)
  source_features=paste0(feature1,"&",feature2)
  cv_op=data.frame(name=source_features, iv=IV, iv_f1=iv_feature1, iv_f2=iv_feature2, 
                   cor_feature1=cor(eval(parse(text=paste0("data$",feature1,"_",feature2))),feature1_vec),
                   cor_feature2=cor(eval(parse(text=paste0("data$",feature1,"_",feature2))),feature2_vec))
  return(cv_op)
}
cv_final_div=NULL

for (i in (1:nrow(class_num))){
  for (j in (1:nrow(class_num))){
    if (j<=i){next}
    cv_final_div=rbind(cv_final_div,select_features_exp(data=train, feature1=class_num$features[i], 
                                                feature2=class_num$features[j],"/"))
  }
}

write.csv(cv_final_div,"cv_final_num_div.csv")

cv_final_div=NULL
for (i in (1:nrow(class_num))){
  for (j in (1:nrow(class_num))){
    if (j<=i){next}
    cv_final_div=rbind(cv_final_div,select_features_exp(data=train, feature1=class_num$features[i], 
                                                        feature2=class_num$features[j],"*"))
  }
}

write.csv(cv_final_div,"cv_final_num_mul.csv")

cv_final_div=NULL
for (i in (1:nrow(class_num))){
  for (j in (1:nrow(class_num))){
    if (j<=i){next}
    cv_final_div=rbind(cv_final_div,select_features_exp(data=train, feature1=class_num$features[i], 
                                                        feature2=class_num$features[j],"+"))
  }
}
write.csv(cv_final_div,"cv_final_num_add.csv")

cv_final_div=NULL
for (i in (1:nrow(class_num))){
  for (j in (1:nrow(class_num))){
    if (j<=i){next}
    cv_final_div=rbind(cv_final_div,select_features_exp(data=train, feature1=class_num$features[i], 
                                                        feature2=class_num$features[j],"-"))
  }
}
write.csv(cv_final_div,"cv_final_num_sub.csv")