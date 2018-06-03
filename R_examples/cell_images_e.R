# http://curso-r.com/blog/2017/06/08/2017-06-08-keras-no-ubuntu/
# https://tensorflow.rstudio.com/keras/
devtools::install_github("rstudio/keras", force =TRUE)

install.packages("keras")
install.packages("tensorflow")
install.packages("curl")
library(curl)
library(keras)
library(tensorflow)

install_keras()

## Cargar y pre-procesar imágenes

cell_images <- read.csv('/home/stiago/Descargas/dataset2-master/dataset2-master/labels.csv')
head(cell_images)
#   X Image   Category
#1 NA     0 NEUTROPHIL
#2 NA     1 NEUTROPHIL
#3 NA     2 NEUTROPHIL
#4 NA     3 NEUTROPHIL
#5 NA     4 NEUTROPHIL
#6 NA     5 NEUTROPHIL

str(cell_images)
#'data.frame':	411 obs. of  3 variables:
# $ X       : logi  NA NA NA NA NA NA ...
# $ Image   : int  0 1 2 3 4 5 6 7 8 9 ...
# $ Category: Factor w/ 18 levels "","BASOPHIL",..: 11 11 11 11 11 11 11 11 2 3 ...

dim(cell_images)
#[1] 411   3

plot(cell_images$Image,
     cell_images$Category,
     pch=21, bg=c("red","green3","blue", "orange")[unclass(cell_images$Category)],
     xlab="a",
     ylab="b")

summary(cell_images)
#          X               Image             Category
#       Mode:logical   Min.   :  0.0   NEUTROPHIL:207
#       NA's:411       1st Qu.:102.5   EOSINOPHIL: 88
#                      Median :205.0             : 44
#                      Mean   :205.0   LYMPHOCYTE: 33
#                      3rd Qu.:307.5   MONOCYTE  : 21
#                      Max.   :410.0   BASOPHIL  :  3
#                                      (Other)   : 15

train_dir <- '/home/stiago/Descargas/dataset2-master/dataset2-master/images/TRAIN/'
validation_dir <- '/home/stiago/Descargas/dataset2-master/dataset2-master/images/TEST_SIMPLE/'
test_dir <- '/home/stiago/Descargas/dataset2-master/dataset2-master/images/TEST/'

img_sample <- image_load(path = '/home/stiago/Descargas/dataset2-master/dataset2-master/images/TRAIN/EOSINOPHIL/_9_9893.jpeg', target_size = c(150, 150))
img_sample_array <- array_reshape(image_to_array(img_sample), c(1, 150, 150, 3))
plot(as.raster(img_sample_array[1,,,] / 255))

# https://tensorflow.rstudio.com/keras/reference/image_data_generator.html
train_datagen      <- image_data_generator(rescale = 1/255)
validation_datagen <- image_data_generator(rescale = 1/255)
test_datagen <- image_data_generator(rescale = 1/255)

# https://tensorflow.rstudio.com/keras/reference/flow_images_from_directory.html
train_data <- flow_images_from_directory(
  directory = train_dir,
  generator = train_datagen,
  target_size = c(150, 150),   # (w, h) --> (150, 150)
  batch_size = 20,             # grupos de 20 imágenes
  class_mode = "binary" #"binary"        # etiquetas binarias
)
# El resultado es : Found 9957 images belonging to 4 classes.

validation_data <- flow_images_from_directory(
  directory = validation_dir,
  generator = validation_datagen,
  target_size = c(150, 150),   # (w, h) --> (150, 150)
  batch_size = 20,             # grupos de 20 imágenes
  class_mode = "binary" #"binary"        # etiquetas binarias
)
# El resultado es: Found 71 images belonging to 4 classes.

test_data <- flow_images_from_directory(
  directory = test_dir,
  generator = test_datagen,
  target_size = c(150, 150),   # (w, h) --> (150, 150)
  batch_size = 20,             # grupos de 20 imágenes
  class_mode = "binary" #"binary"        # etiquetas binarias
)
# El resultado es: Found 2487 images belonging to 4 classes.

# To categorical
# train_data <- to_categorical(train_data, NULL)
# test_data <- to_categorical(test_data)#, 4)
# validation_data <- to_categorical(validation_data, 4)

## -------------------------------------------------------------------------------------
## Crear modelo

# Definir arquitectura
# https://tensorflow.rstudio.com/keras/articles/sequential_model.html
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32,  kernel_size = c(3, 3), activation = "relu", input_shape = c(150, 150, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 32,  kernel_size = c(3, 3), activation = "relu", input_shape = c(150, 150, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64,  kernel_size = c(3, 3), activation = "relu") %>% layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  #layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>% layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  #layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>% layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dropout(rate = 0.7) %>%
  #layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

summary(model)
# result in picture 1

# Compilar modelo
# https://tensorflow.rstudio.com/keras/reference/compile.html
model %>% compile(
  loss= "poisson", #'categorical_crossentropy', #optimizer='adam',
  #loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(), #(lr = 1e-4),
  metrics = c("accuracy")
)

# Entrenamiento
# https://tensorflow.rstudio.com/keras/reference/fit_generator.html
history <- model %>%
  fit_generator(
    train_data,
    steps_per_epoch = 100,
    epochs = 5,
    validation_data = validation_data,
    validation_steps = 50
  )

# Visualizar entrenamiento
plot(history)

# Guardar modelo (HDF5)
# https://tensorflow.rstudio.com/keras/reference/save_model_hdf5.html
model %>% save_model_hdf5("cells_types.h5")

# Evaluar modelo
# https://tensorflow.rstudio.com/keras/reference/evaluate_generator.html
model %>% evaluate_generator(test_data, steps = 50)
# $loss
# -8.178444
#$acc
# 0.228


## -------------------------------------------------------------------------------------
## Data augmentation

# https://tensorflow.rstudio.com/keras/reference/image_data_generator.html
data_augmentation_datagen <- image_data_generator(
  rescale = 1/255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE,
  fill_mode = "nearest"
)

train_augmented_data <- flow_images_from_directory(
  directory = train_dir,
  generator = data_augmentation_datagen,  # ¡usando nuevo datagen!
  target_size = c(150, 150),   # (w, h) --> (150, 150)
  batch_size = 20,             # grupos de 20 imágenes
  class_mode = "binary"        # etiquetas binarias
)

history_augmentation <- model %>%
  fit_generator(
    train_augmented_data,
    steps_per_epoch = 100,
    epochs = 15,
    validation_data = validation_data,
    validation_steps = 50
  )


plot(history_augmentation)

model %>% save_model_hdf5("cells_types_augmentation.h5")

model %>% evaluate_generator(test_data, steps = 50)