from basic_libs import *


def fine_tune_model(X_train, X_test, y_train, y_test, num_classes,base_model):
    
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )

    # Train the model
  #  model.fit(datagen.flow(X_train, y_train, batch_size=32),
   #           validation_data=(X_test, y_test),
    #          epochs=5)

    for layer in model.layers[:2]:  
        layer.trainable = True

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(datagen.flow(X_train, y_train, batch_size=32),validation_data=(X_test, y_test),epochs=2)

    feature_extractor = Model(inputs=model.input, outputs=base_model.output)
    X_train_features = feature_extractor.predict(X_train)
    X_test_features = feature_extractor.predict(X_test)

    X_train_features_flat = X_train_features.reshape(X_train_features.shape[0], -1)
    X_test_features_flat = X_test_features.reshape(X_test_features.shape[0], -1)

    #    return X_train_features_flat, X_test_features_flat, y_train, y_test
    X=np.concatenate((X_train_features_flat, X_test_features_flat), axis=1)
    y=np.concatenate((y_train, y_test), axis=1)
    
    return X_train_features_flat, X_test_features_flat, y_train, y_test
