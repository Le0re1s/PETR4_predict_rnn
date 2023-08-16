################################################################################
#                  Prevendo valores das ações da Petrobras (PETR4)             #     
#                  com base em uma Recurrent Neural Networks (rnn)             #
################################################################################

library("rnn")
library("dplyr")

data <- read.csv("PETR4.SA.csv")

fechamento <- data$Close

fecha_dia_anterior <- lead(fechamento,n=1L)

data_analise <- data.frame(fechamento)
data_analise$fecha_dia_anterior <- fecha_dia_anterior

# Estatísticas
summary(data_analise)

# Excluindo os NAs
data_analise <- data_analise[1:248,]

# Separei a target (y) e a feature (x)
x <- data_analise[,2]
y <- data_analise[,1]

# Recorrências em 3 dimensões (facilitar o processamento por partes):
# 1: amostra
# 2: quantas vezes vou passar a amostra na rede
# 3: número de features
X <- matrix(x, nrow = 31)
Y <- matrix(y, nrow = 31)

# Colocando valores de 0 a 1 pra melhorar a performance
Yscaled <- (Y - min(Y)) / (max(Y) - min(Y))
Xscaled <- (X - min(X)) / (max(X) - min(X))
Y <- Yscaled
X <- Xscaled

train=1:6
test=7:8

set.seed(123)
model <- trainr(Y = Y[,train],
                X = X[,train],
                learningrate = 0.05,
                hidden_dim = 20,
                numepochs = 1000,
                network_type = "rnn"
)

# Prevendo com a base train
Ytrain <- t(matrix(predictr(model, X[,train]),nrow=1))
Yreal <- t(matrix(Y[,train],nrow=1))

# Percentual da correlação entre matrizes
rsq <- function(y_actual,y_predict){
  cor(y_actual,y_predict)^2
}

rsq(Yreal,Ytrain)

plot(Ytrain, col = 'red', type='l',
     main = "Prevendo valor das ações: PETR4",
     ylab = "Treino, Real")
lines(Yreal, type = 'l', col = 'darkblue')
legend("bottomright", c("Previsão", "Atual"),
       col = c("red","darkblue"),
       lty = c(1,1), lwd = c(1,1))

# Já na base test...
Ytest = t(matrix(Y[,test], nrow = 1))
Ypredicted = t(matrix(predictr(model, Y[,test]), nrow = 1))

result_data <- data.frame(Ytest)
result_data$Ypredicted <- Ypredicted     

# Correlação final
rsq(result_data$Ytest,result_data$Ypredicted)

mean(result_data$Ytest)
mean(result_data$Ypredicted)