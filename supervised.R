#call libraries
library (ISLR)
library(ggpubr)
library(bestNormalize)
library(e1071)
library(Hmisc)
library(car)
library(boot)
library(corrplot)
library(MASS)
library(glmnet)
library(tree)
library(rpart)
library(rpart.plot)
library(randomForest)
library(caTools)

#choose dataset
fix(College)
names(College)
College= na.omit(College)
c = c(1,2,5,6,9,10,11,12,17)
College= College [,c ]

#exploratory analysis
par(mfrow=c(2,4))
for (i in c(2,3,4,5,6,7,8)){
  boxplot(College[,i], data = College, xlab=colnames(College)[i])
}
summary(College$Expend)
ggdensity(College, x = "Expend", fill = "lightgray", title = "Expend") +
  stat_overlay_normal_density(color = "red", linetype = "dashed")

#pick the best normalization process automatically
(BNobject <- bestNormalize(College$Expend))
par(mfrow = c(1,2))
MASS::truehist(BNobject$x.t, 
               main = paste("Best Transformation:", 
                            class(BNobject$chosen_transform)[1]), nbins = 12)
College$normExpend <-BNobject$x.t
skewness(College$normExpend, na.rm = TRUE)
ggdensity(College, x = "normExpend", fill = "lightgray", title = "normExpend") +
  stat_overlay_normal_density(color = "red", linetype = "dashed")

#splitting between train set and test set
set.seed(123)
row.number = sample(1:nrow(College),0.5*nrow(College))
train= College[row.number,]
test= College[-row.number,]

#fit the multiple linear regression model
College$Private <- ifelse(College$Private == 'Yes', 1, 0)
full.model <- lm(normExpend ~ .-Expend, data = train)
summary(full.model)
par(mfrow=c(2,2))
plot(full.model)
plot(hatvalues(full.model))

#check for collinearity issue:
vif(full.model) 
sqrt(vif(full.model)) > 2 

#correlation matrix
a<-c(2,3,4,5,6,7,8)
res<-cor(College[a]) 
round(res, 2)
symnum(res, abbr.colnames = FALSE)
rcorr(as.matrix(College[a]))
flattenCorrMatrix <- function(cormat, pmat) {
  ut <- upper.tri(cormat)
  data.frame(
    row = rownames(cormat)[row(cormat)[ut]],
    column = rownames(cormat)[col(cormat)[ut]],
    cor  =(cormat)[ut],
    p = pmat[ut]
  )
}
res1<-rcorr(as.matrix(College[a]))
flattenCorrMatrix(res1$r, res1$P)
col<- colorRampPalette(c("blue", "white", "red"))(20)
heatmap(x = res, col = col, symm = TRUE, Colv = NA, Rowv = NA)
corrplot(res, type = "upper", 
         tl.col = "black", tl.srt = 45)

#fit the multiple linear regression model without collinearity
lm.fit <- lm(normExpend ~ Private+ Apps+Top10perc+ Outstate+ Room.Board+ Books+Personal, data = train)
summary(lm.fit)

#fit the multiple linear regression model without useless variables
lm.fit1 <- lm(normExpend ~ Top10perc+ Outstate+ Room.Board, data = train)
summary(lm.fit1)

#prediction
pred_test = predict(lm.fit1, newdata = test)
mse = sum((pred_test - test$normExpend)^2)/length(test$normExpend)
c(MSE_test = mse, R2=summary(lm.fit1)$r.squared)
pred_train= predict(lm.fit1, newdata=train)
mse= sum((pred_train-train$normExpend)^2)/length(train$normExpend)
c(MSE_train=mse, R2=summary(lm.fit1)$r.squared)
par(mfrow=c(1,1))
ggplot(test, aes(x = normExpend, y = pred_test))+
  geom_point() +
  geom_smooth(method = 'lm', se = FALSE, col= 'red') +              
  theme_bw()       # Plotting of the predicted response variable using test set
ggplot(train, aes(x = normExpend, y = pred_train)) +
  geom_point() +
  geom_smooth(method = 'lm', se = FALSE) +
  theme_bw()      # Plotting of the predicted response variable using training set 

#predictors plotting
par(mfrow=c(2,2))
plot(College$Private,College$normExpend)
plot(College$Apps,College$normExpend)
plot(College$Top10perc,College$normExpend)
plot(College$Top25perc,College$normExpend)
plot(College$Outstate,College$normExpend)
plot(College$Room.Board,College$normExpend)
plot(College$Books,College$normExpend)
plot(College$Personal,College$normExpend)

#polynomial regression
plot (College$normExpend, College$Top10perc)
lin <- lm(normExpend ~ Top10perc+ Outstate+ Room.Board, data = train)
pred_test = predict(lin, newdata = test)
mse = sum((pred_test - test$normExpend)^2)/length(test$normExpend)
c(MSE_test = mse, R2=summary(lin)$r.squared)
pol_mod2 <- lm(normExpend ~poly(Top10perc, 2)+ Outstate + Room.Board, data = train)
pred_test = predict(pol_mod2, newdata = test)
mse = sum((pred_test - test$normExpend)^2)/length(test$normExpend)
c(MSE_test = mse, R2=summary(pol_mod2)$r.squared)
pol_mod3 <- lm(normExpend ~poly(Top10perc,3)+ Outstate + Room.Board, data = train)
pred_test = predict(pol_mod3, newdata = test)
mse = sum((pred_test - test$normExpend)^2)/length(test$normExpend)
c(MSE_test = mse, R2=summary(pol_mod3)$r.squared)

#k fold cross validation
set.seed(100)
cv.error.10 = rep(0,10)
for (i in 1:10){
  glm.fit = glm (normExpend ~poly(Top10perc,i )+ Outstate + Room.Board, data = College)
  cv.error.10[i]= cv.glm (College,glm.fit, K= 10)$delta[1]
}
cv.error.10

#ANOVA
anova(full.model, pol_mod2)

#stepwise selection
step <- stepAIC(pol_mod3, direction="both")
step$anova # display results
summary(step)
par(mfrow=c(2,2))
plot(step)

#shrinkage methods
#ridge Regression (alpha = 0)
x= model.matrix(normExpend~.-Expend,College)[,-1]
y= College$normExpend
set.seed(333)
train=sample(1:nrow(x),nrow(x)/2)
test= (-train)
y.test=y[test]
# CV for tuning parameter lambda
set.seed(111)
cv.out= cv.glmnet(x[train,],y[train],alpha=0,standardize=TRUE)
plot(cv.out,col= 'red')
bestlam=cv.out$lambda.min
c(`Best Lambda`= bestlam)
# MSE of Ridge Regression using optimal lambda
grid= 10^seq(10,-2,length=100)
ridge.mod= glmnet(x[train,],y[train],alpha=0,
                  lambda=grid,thresh=1e-12,standardized= TRUE)
ridge.pred= predict(ridge.mod, s=bestlam, newx=x[test,])
c(MSE= mean((ridge.pred-y.test)^2))
out= glmnet(x,y,alpha=0)
ridge.coef= predict(out,type = "coefficients", s=bestlam)[1:9,]
as.table(ridge.coef)
plot(ridge.mod,xvar='lambda',label=TRUE)
title(main= 'Ridge', line = 2.4)

#lasso regression (alpha=1)
# CV for tuning lambda parameter for lasso regression
set.seed(121)
cv.lasso=cv.glmnet(x[train,],y[train],alpha=1)
plot(cv.lasso,col='red')
bestlam=cv.lasso$lambda.1se
c(`Best Lambda Lasso`=bestlam) 
lasso.mod= glmnet(x[train,],y[train],alpha= 1,
                  lambda=grid,thresh=1e-12,standardized= TRUE)
lasso.pred=predict(lasso.mod,s=bestlam,newx=x[test,])
c(`MSE Lasso`=mean((lasso.pred-y.test)^2))

#final model with Lasso Regression
out=glmnet(x,y,alpha=1,lambda=grid)
lasso.coef= predict(out, type="coefficients",s=bestlam)[1:9,]
as.table(lasso.coef)
plot(lasso.mod,xvar='lambda',label=TRUE)
title(main= 'Lasso', line = 2.4)

#tree-Based Methods
#regression tree
set.seed(111)
split = sample.split(College$normExpend, SplitRatio = 0.5)
train = subset(College, split==TRUE, standardized= TRUE)
test = subset(College, split==FALSE, standardized = TRUE)
tree.dataset<- tree(normExpend ~.-Expend, data=train)
plot (tree.dataset)
text(tree.dataset, pretty= 0, digits=3, cex = 0.75)
title(main='Regression Tree')
tree.r = rpart(normExpend~.-Expend, data=train)
prp(tree.r)
summary(tree.r)
printcp(tree.r)
plotcp(tree.r)

#accuracy of base model regression tree (MSE and RMSE)
tree.pred= predict(tree.r, newdata=test)
tree.mse= mean((tree.pred-test$normExpend)^2)
tree.rmse= sqrt(mean((tree.pred-test$normExpend)^2))
c(MSE= tree.mse)
c(RMSE= tree.rmse)

#pruning tree with cp optimal
tree.pruned = prune(tree.r, cp= 0.039)
prp(tree.pruned)

#accuracy of the pruned regression tree
pred.pruned = predict(tree.pruned, test)
pruned.mse= mean((pred.pruned-test$normExpend)^2) 
pruned.rmse= sqrt(mean((pred.pruned-test$normExpend)^2)) 
c(Pr.MSE= pruned.mse)
c(Pr.RMSE= pruned.rmse)

#random forest
rf_tree <- randomForest(normExpend ~.-Expend, data=train,
                        mtry=3, importance = TRUE, ntree=100)

#accuracy of the random forest
pred.rf = predict(rf_tree , newdata=test)
mse.rf= mean((pred.rf-test$normExpend)^2) 
c('MSE random forest' = mse.rf)

#variable importance measure
importance(rf_tree)
varImpPlot(rf_tree, main = 'Random Forest - Variable Importance')

