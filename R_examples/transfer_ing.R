#Mas pruebas 

## Extracción de características
# Cargar capa convolutiva de VGG16, pre-entrenada con ImageNet
conv_base <- application_vgg16(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(150, 150, 3)
)

# Crear función que extrae feature map --con dimension (4, 4, 512) para VGG16
batch_size <- 20
extract_features <- function(directory, sample_count) {
  
  # crear generador para transformación de imágenes de entrada
  datagen <- image_data_generator(rescale = 1/255)
  
  # crear arrays de salida, inicialmente a 0
  #  features: características (números de samples x (4, 4, 512) )
  #  labels: (número de samples)
  features <- array(0, dim = c(sample_count, 4, 4, 512))
  labels <- array(0, dim = c(sample_count))
  
  # leer de directorio pasado como parámetro
  data_generator <- flow_images_from_directory(
    directory = directory,
    generator = datagen,
    target_size = c(150, 150),
    batch_size = batch_size,
    class_mode = "binary"
  )
  
  # extraer batches hasta acumular el número de samples pasado como parámetro
  i <- 0
  while(TRUE) {
    batch <- generator_next(data_generator)
    inputs_batch <- batch[[1]]
    labels_batch <- batch[[2]]
    features_batch <- conv_base %>% predict(inputs_batch)
    index_range <- ((i * batch_size)+1):((i + 1) * batch_size)
    features[index_range,,,] <- features_batch
    labels[index_range] <- labels_batch
    i <- i + 1
    if (i * batch_size >= sample_count)
      break 
  }
  
  # devolver feature map y labels
  list(
    features = features,
    labels = labels
  ) 
}

# Generar conjuntos de entrenamiento, validación y test con características extraídas
train      <- extract_features(train_dir, 2000)
validation <- extract_features(validation_dir, 1000) #100 #200
test       <- extract_features(test_dir, 1000)

# Redimensionar datos de los feature maps
reshape_features <- function(features) {
  array_reshape(features, dim = c(nrow(features), 4 * 4 * 512))
}
train$features      <- reshape_features(train$features)
validation$features <- reshape_features(validation$features)
test$features       <- reshape_features(test$features)

# Crear clasificador con datos de los feature maps (red neuronal)
model <- keras_model_sequential() %>%
  layer_dense(units = 256, activation = "relu", input_shape = 4 * 4 * 512) %>%
  layer_dense(units = 1, activation = "softmax")

model %>% compile(
  optimizer = optimizer_rmsprop(lr = 2e-5),
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

history <- model %>% 
  fit(
    train$features, train$labels,
    epochs = 12,
    batch_size = 20,
    validation_data = list(validation$features, validation$labels)
  )

plot(history)

model %>% save_model_hdf5("cells_feature-extraction.h5")

model %>% evaluate(test$features, test$labels)




## Fine tuning (solo con GPU)

# 1. Crear modelo completo utilizando la capa convolutiva de VGG16 pre-entrenada y nuestra capa FC
model <- keras_model_sequential() %>%
  conv_base %>%
  layer_flatten() %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 4, activation = "softmax")

# 2. Congelar pesos de la capa convolutiva VGG16
freeze_weights(conv_base)

# 3. Entrenamiento 'end-to-end' (pero solo se modifica de la capa FC)
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(lr = 2e-4),
  metrics = c("accuracy")
)

history <- model %>% 
  fit_generator(
    train_data,
    steps_per_epoch = 100,
    epochs = 5,
    validation_data = validation_data,
    validation_steps = 50
  )

# 4. Descongelar pesos de las capas de las red base
unfreeze_weights(conv_base, from = "block3_conv1")

# 5. Entrenar capa descongelada y FC
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-4),
  metrics = c("accuracy")
)

history <- model %>% 
  fit_generator(
    train_data,
    steps_per_epoch = 150,
    epochs = 5,
    validation_data = validation_data,
    validation_steps = 50
  )

plot(history)

model %>% save_model_hdf5("cells-tuning.h5")

model %>% evaluate_generator(test_data, steps = 50)
