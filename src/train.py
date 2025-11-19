import argparse
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from model import build_model

def get_datasets(data_dir, img_size=(224,224), batch_size=32, val_split=0.2, seed=123):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset='training',
        seed=seed,
        image_size=img_size,
        batch_size=batch_size
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset='validation',
        seed=seed,
        image_size=img_size,
        batch_size=batch_size
    )
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/train')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()
    train_ds, val_ds = get_datasets(args.data_dir, batch_size=args.batch_size)
    class_names = train_ds.class_names
    num_classes = len(class_names)
    model = build_model(num_classes=num_classes)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
        ModelCheckpoint('experiments/best_model.h5', save_best_only=True, monitor='val_accuracy')
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)
    model.save('experiments/final_model')
    with open('experiments/class_names.txt','w') as f:
        for c in class_names:
            f.write(c+'\n')

if __name__ == '__main__':
    main()
