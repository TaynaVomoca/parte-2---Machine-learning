###############################
#Pacotes
###############################
install.packages("tm") #tratar dados de texto
install.packages("SnowballC")
install.packages("wordcloud")
install.packages("e1071")
install.packages("caret") #matrix de confusão

library(tm)
library(SnowballC)
library(wordcloud)
library(e1071)
library(caret)

################################################################
# Parte 1 - criar um corpus (conjunto de documentos de texto)
################################################################

sms_corpus <- VCorpus(VectorSource(Exemplo_2_Dados_sms$text))

############################################
# Parte 2 - Limpar os documentos de texto
############################################

# 1 - transformar tudo em letra minúscula
sms_limpo <- tm_map(sms_corpus, content_transformer(tolower))

# Comparando
as.character(sms_corpus[[1]])
as.character(sms_limpo[[1]])

# 2 - Remover números
sms_limpo <- tm_map(sms_limpo, removeNumbers)

# 3 - Remover stopwords
sms_limpo <- tm_map(sms_limpo, removeWords, stopwords()) #(kind = "pt - br")) dentro de stopword

# 4 - Remover pontuações
sms_limpo <- tm_map(sms_limpo, removePunctuation)

# 5 - Remover espaços em excesso
sms_limpo <- tm_map(sms_limpo, stripWhitespace)

# 6 - Manter os radicais
sms_limpo <- tm_map(sms_limpo, stemDocument)

##########################
# Parte 3 - Tokenização
##########################

# Criando a matriz de termos dos documentos(DTM)
sms_dtm <- DocumentTermMatrix(sms_limpo)

#Separando os dados em treino e teste
sms_dtm_treino <- sms_dtm[1:4169,]
sms_dtm_teste <- sms_dtm[4170:5572,]

sms_dtm_rotulos_treino <- Exemplo_2_Dados_sms[1:4169,]$type
sms_dtm_rotulos_teste <- Exemplo_2_Dados_sms[4170:5572,]$type

# Visualização em nuvem de palavras
wordcloud(sms_limpo, min.freq = 50, random.order = FALSE)

spam <- subset(Exemplo_2_Dados_sms, type = "spam")
wordcloud(spam$text, min.freq = 40, random.order = FALSE, scale = c(3, 0.5))

ham <- subset(Exemplo_2_Dados_sms, type = "ham")
wordcloud(ham$text, min.freq = 40, random.order = FALSE, scale = c(3, 0.5))

# Deixando apenas palavras com frequencia >= 5
sms_palavras_freq_treino <- findFreqTerms(sms_dtm_treino, 5)
sms_palavras_freq_teste <- findFreqTerms(sms_dtm_teste, 5)

sms_dtm_freq_treino <- sms_dtm_treino[, sms_palavras_freq_treino]
sms_dtm_freq_teste <- sms_dtm_teste[, sms_palavras_freq_teste]

# Finalmente, contando as palavras em cada mensagem
converter_contagens <- function(x) {
  x <- ifelse(x > 0, "Sim", "Não")
}

sms_treino <- apply(sms_dtm_freq_treino, MARGIN = 2, converter_contagens)
sms_teste <- apply(sms_dtm_freq_teste, MARGIN = 2, converter_contagens)

###########################
# Parte 4 - Classificador
###########################

#treino
sms_classificador <- naiveBayes(sms_treino, sms_dtm_rotulos_treino)

#teste
sms_teste_pred <- predict(sms_classificador, sms_teste)

confusionMatrix(sms_teste_pred, factor(sms_dtm_rotulos_teste))




