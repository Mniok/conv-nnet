print("uruchomiono program")
#1 - importowanie bibliotek
import tensorflow

from tensorflow.keras import datasets, layers, models, utils, preprocessing
import matplotlib.pyplot as plt
import tensorflow.keras.preprocessing.image as kerasImg
import numpy
print("zaimportowano biblioteki")




#2 - przygotowanie danych

rozmiarObrazow = (64, 64)

zbiorObrazow = []
zbiorLabels = []
zbiorValidationObrazow = []
zbiorValidationLabels = []
klasy = ["nie ma krzesla", "jest krzeslo"]
lr=0.0003 #współczynnik uczenia
mc=0.9 #momentum constant
liczbaEpok = 200

sciezka = "obrazy\\jest krzeslo\\"
for i in range(40):
    sciezkaObrazu = sciezka + format(i+1, "02d") + ".jpg"
    #format: https://docs.python.org/2/library/string.html#format-specification-mini-language
    #02d - leading zeros, length 2, decimal
    obraz = kerasImg.load_img(sciezkaObrazu,
        color_mode="rgb", #kolor obrazu zrodlowego
        target_size=rozmiarObrazow
    )
    dane = tensorflow.image.rgb_to_grayscale(obraz)
    #rgb_to_grayscale zastepuje dodatkowo kerasImg.img_to_array()
    zbiorObrazow.append(dane) #tf.Tensor danych
    zbiorLabels.append([1]) # [1] bo labelsy w Keras sa arrayami 1-elementowymi

sciezka = "obrazy\\nie ma krzesla\\"
for i in range(40):
    sciezkaObrazu = sciezka + format(i+1, "02d") + ".jpg"
    obraz = kerasImg.load_img(sciezkaObrazu,
        color_mode="rgb",
        target_size=rozmiarObrazow
    )
    dane = tensorflow.image.rgb_to_grayscale(obraz)
    zbiorObrazow.append(dane)
    zbiorLabels.append([0])



sciezka = "obrazy\\validation jest krzeslo\\"
for i in range(10):
    sciezkaObrazu = sciezka + format(i+1, "02d") + ".jpg"
    obraz = kerasImg.load_img(sciezkaObrazu,
        color_mode="rgb",
        target_size=rozmiarObrazow
    )
    dane = tensorflow.image.rgb_to_grayscale(obraz)
    zbiorValidationObrazow.append(dane)
    zbiorValidationLabels.append([1])

sciezka = "obrazy\\validation nie ma krzesla\\"
for i in range(10):
    sciezkaObrazu = sciezka + format(i+1, "02d") + ".jpg"
    obraz = kerasImg.load_img(sciezkaObrazu,
        color_mode="rgb",
        target_size=rozmiarObrazow
    )
    dane = tensorflow.image.rgb_to_grayscale(obraz)
    zbiorValidationObrazow.append(dane)
    zbiorValidationLabels.append([0])

print("wczytano dane")

#3.2 pomieszanie danych uczacych
zbiorShuffle = []
for i in range(len(zbiorObrazow)):
	zbiorShuffle += [[zbiorObrazow[i], zbiorLabels[i]]]
	
numpy.random.shuffle(zbiorShuffle)

for i in range(len(zbiorShuffle)):
	zbiorObrazow[i] = zbiorShuffle[i][0]
	zbiorLabels[i] = zbiorShuffle[i][1]


#3.3 - weryfikacja danych 
plt.figure(figsize=(8,8))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(zbiorObrazow[i], cmap=plt.cm.gray) #wyswietla obraz w skali szarosci
    plt.xlabel(klasy[zbiorLabels[i][0]]) #labelsy sa arrayami 1-elem
plt.show()


#3.4 normalizacja danych
for obr in range(len(zbiorObrazow)):
    zbiorObrazow[obr] = zbiorObrazow[obr].numpy().tolist()
    #konwersja tensor -> numpy array -> array
    #normalizacja:
    for i in range(len(zbiorObrazow[obr])):
        for j in range(len(zbiorObrazow[obr][i])):
            zbiorObrazow[obr][i][j][0] = zbiorObrazow[obr][i][j][0] / 255

for obr in range(len(zbiorValidationObrazow)):
    zbiorValidationObrazow[obr] = zbiorValidationObrazow[obr].numpy().tolist()
    for i in range(len(zbiorValidationObrazow[obr])):
        for j in range(len(zbiorValidationObrazow[obr][i])):
            zbiorValidationObrazow[obr][i][j][0] = zbiorValidationObrazow[obr][i][j][0] / 255


#4 - budowa modelu
print("#4 - budowa modelu")
model = models.Sequential()
model.add(layers.Conv2D(14, (3, 3), activation='relu', input_shape=(64, 64, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(11, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(11, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(24, activation='relu'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1))

#model.summary()

#5 - kompilacja i uczenie modelu
print("#5 - kompilacja i uczenie modelu")
optimizerMomentum = tensorflow.keras.optimizers.SGD(
    learning_rate=lr, momentum=mc)
model.compile(optimizer=optimizerMomentum,
              loss=tensorflow.keras.losses.BinaryCrossentropy(),
              metrics=[tensorflow.keras.metrics.BinaryAccuracy()])

history = model.fit(zbiorObrazow,
                    zbiorLabels,
                    epochs=liczbaEpok,
                    validation_data=(zbiorValidationObrazow,
                                     zbiorValidationLabels))



#6 - wykresy
acc = history.history['binary_accuracy']
val_acc = history.history['val_binary_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(liczbaEpok)

#   6.2 wygładzenie
smooth_size = 10
acc_smooth = []
val_acc_smooth = []

for i in range(smooth_size):
    acc_smooth += [acc[i]]
    val_acc_smooth += [val_acc[i]]

for i in range(smooth_size, len(acc)):
    smoothen = 0
    for s in range(smooth_size):
        smoothen += acc[i-s]
    acc_smooth += [smoothen/smooth_size]
    smoothen = 0
    for s in range(smooth_size):
        smoothen += val_acc[i-s]
    val_acc_smooth += [smoothen/smooth_size]

#   6.3 rysowanie wykresów trafności i błędu
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc_smooth, label='Trafność na zbiorze uczącym')
plt.plot(epochs_range, val_acc_smooth, label='Trafność na zbiorze walidacyjnym')
plt.legend(loc='lower right')
plt.xlabel("Epoka")
plt.ylabel("Trafność")
plt.title('Wykres trafności')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Błąd na zbiorze uczącym')
plt.plot(epochs_range, val_loss, label='Błąd na zbiorze walidacyjnym')
plt.legend(loc='upper right')
plt.xlabel("Epoka")
plt.ylabel("Błąd")
plt.title('Wykres funkcji błędu')
plt.show()

print("końcowe:")
print("[niewygł]Trafność na zbiorze uczącym", acc[len(acc)-1])
print("[wygł]Trafność na zbiorze uczącym", acc_smooth[len(acc_smooth)-1])
print("[niewygł]Trafność na zbiorze walidacyjnym", val_acc[len(val_acc)-1])
print("[wygł]Trafność na zbiorze walidacyjnym", val_acc_smooth[len(val_acc_smooth)-1])
print("Loss na zbiorze uczącym", loss[len(loss)-1])
print("Loss na zbiorze walidacyjnym", val_loss[len(val_loss)-1])

#history.history


#7. eksperymenty
#   7.1 eksperyment learning rate
print("7.1 eksperyment learning rate")
liczbaEpok = 100
lr_range = []
for i in range(9):
    lr_range += [(i+1)/100000]
for i in range(9):
    lr_range += [(i+1)/10000]
for i in range(9):
    lr_range += [(i+1)/1000]
for i in range(9):
    lr_range += [(i+1)/100]
for i in range(10):
    lr_range += [(i+1)/10]

eksperymentlr_acc = []
eksperymentlr_val_acc = []
for lr in lr_range:
    optimizerMomentum = tensorflow.keras.optimizers.SGD(
    learning_rate=lr, momentum=mc)
    model.compile(optimizer=optimizerMomentum,
                  loss=tensorflow.keras.losses.BinaryCrossentropy(),
                  metrics=[tensorflow.keras.metrics.BinaryAccuracy()])

    history = model.fit(zbiorObrazow,
                        zbiorLabels,
                        epochs=liczbaEpok,
                        validation_data=(zbiorValidationObrazow,
                                         zbiorValidationLabels),
                        verbose=0) #nie wypisuje wynikow co epokę

    acc = history.history['binary_accuracy']
    val_acc = history.history['val_binary_accuracy']

    eksperymentlr_acc += [acc[len(acc)-1]]
    eksperymentlr_val_acc += [val_acc[len(val_acc)-1]]
    print("dla lr =", lr, "osiagnieta trafność =", [acc[len(acc)-1]])


plt.figure(figsize=(8, 8))
plt.xscale("log")
plt.plot(lr_range, eksperymentlr_acc, label='Trafność na zbiorze uczącym')
plt.plot(lr_range, eksperymentlr_val_acc, label='Trafność na zbiorze walidacyjnym')
plt.legend(loc='lower right')
plt.xlabel("Współczynnik uczenia")
plt.ylabel("Trafność")
plt.title('Zależność trafności od współczynnika uczenia')
plt.show()


#   7.2 eksperyment momentum constant
print("7.2 eksperyment momentum constant")
lr=0.0003
liczbaEpok = 100
mc_range = []
for i in range(20):
    mc_range += [(i+1)/20]


eksperymentmc_acc = []
eksperymentmc_val_acc = []
for mc in mc_range:
    optimizerMomentum = tensorflow.keras.optimizers.SGD(
    learning_rate=lr, momentum=mc)
    model.compile(optimizer=optimizerMomentum,
                  loss=tensorflow.keras.losses.BinaryCrossentropy(),
                  metrics=[tensorflow.keras.metrics.BinaryAccuracy()])

    history = model.fit(zbiorObrazow,
                        zbiorLabels,
                        epochs=liczbaEpok,
                        validation_data=(zbiorValidationObrazow,
                                         zbiorValidationLabels),
                        verbose=0) #nie wypisuje wynikow co epokę

    acc = history.history['binary_accuracy']
    val_acc = history.history['val_binary_accuracy']

    eksperymentmc_acc += [acc[len(acc)-1]]
    eksperymentmc_val_acc += [val_acc[len(val_acc)-1]]
    print("dla mc =", mc, "osiagnieta trafność =", [acc[len(acc)-1]])


plt.figure(figsize=(8, 8))
plt.plot(mc_range, eksperymentmc_acc, label='Trafność na zbiorze uczącym')
plt.plot(mc_range, eksperymentmc_val_acc, label='Trafność na zbiorze walidacyjnym')
plt.legend(loc='lower right')
plt.xlabel("Stała momentum")
plt.ylabel("Trafność")
plt.title('Zależność trafności od stałej momentum')
plt.show()



#   7.3 eksperyment liczby filtrów w warstwach 1, 2 i 3
print("7.3 eksperyment liczby filtrów w warstwach")
first_layer_filters_range = [4, 6, 8, 10, 12, 14, 16, 18, 20]
second_and_3rd_layers_filters_range = [4, 6, 8, 10, 12, 14, 16, 18, 20]
liczbaEpok = 100
lr = 0.0001
mc = 0.9

experiment3_results = []
experiment3_val_results = []

for S1 in first_layer_filters_range:
    res = []
    val_res = []
    #budowa modelu z nową liczbą filtrów
    for S2S3 in second_and_3rd_layers_filters_range:
        model = models.Sequential()
        model.add(layers.Conv2D(S1, (3, 3), activation='relu',
                                input_shape=(64, 64, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(S2S3, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(S2S3, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(1))

        #kompilacja i uczenie
        optimizerMomentum = tensorflow.keras.optimizers.SGD(
            learning_rate=lr, momentum=mc)
        model.compile(optimizer=optimizerMomentum,
                      loss=tensorflow.keras.losses.BinaryCrossentropy(),
                      metrics=[tensorflow.keras.metrics.BinaryAccuracy()])

        history = model.fit(zbiorObrazow,
                            zbiorLabels,
                            epochs=liczbaEpok,
                            validation_data=(zbiorValidationObrazow,
                                             zbiorValidationLabels),
                            verbose=0)

        #zapis wyników
        acc = history.history['binary_accuracy']
        val_acc = history.history['val_binary_accuracy']

        res += [acc[len(acc)-1]]
        val_res += [val_acc[len(val_acc)-1]]
        print("dla S1 =", S1, "i S2S3 = ", S2S3,
              "osiagnieta trafność =", [acc[len(acc)-1]])
    experiment3_results += [res]
    experiment3_val_results += [val_res]


#przygotowanie danych do wykresów
experiment3_results = numpy.array(experiment3_results)
experiment3_val_results = numpy.array(experiment3_val_results)
xaxis, yaxis = numpy.meshgrid(first_layer_filters_range, second_and_3rd_layers_filters_range)


#wykres dla zbioru uczącego
plt.figure(figsize=(8, 8))
trojwym = plt.axes(projection='3d')
trojwym.set_xlabel("Liczba filtrów pierwszej warstwy")
trojwym.set_ylabel("Liczba filtrów drugiej i trzeciej warstwy")
trojwym.set_zlabel("Trafność")
trojwym.plot_surface(xaxis, yaxis,
                     experiment3_results, cmap='jet')
plt.title('Zależność trafności od liczby filtrów na warstwę dla zbioru uczącego')
plt.show()

#wykres dla zbioru walidacyjnego
plt.figure(figsize=(8, 8))
trojwym2 = plt.axes(projection='3d')
trojwym2.set_xlabel("Liczba filtrów pierwszej warstwy")
trojwym2.set_ylabel("Liczba filtrów drugiej i trzeciej warstwy")
trojwym2.set_zlabel("Trafność")
trojwym2.plot_surface(xaxis, yaxis,
                     experiment3_val_results, cmap='plasma')
plt.title('Zależność trafności od liczby filtrów na warstwę dla zbioru walidacyjnego')
plt.show()
