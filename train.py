import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

def compile_model(model):
    optimizer = tf.keras.optimizers.experimental.AdamW(
        learning_rate=0.0001,
        weight_decay=0.004,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        jit_compile=True,
        name="AdamW",
    )
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
    return model

def train_model(model, train_generator, test_generator, checkpoint_path, epochs=150):
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.000001, cooldown=2)

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=test_generator,
        verbose=1,
        callbacks=[learning_rate_reduction, checkpoint]
    )

    return history