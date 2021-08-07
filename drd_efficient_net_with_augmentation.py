from IPython.core.display import display, HTML
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model, Sequential
import numpy as np
import pandas as pd
import shutil
import time
import cv2 as cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import os
import seaborn as sns
sns.set_style('darkgrid')


class LRA(keras.callbacks.Callback):
    reset = False
    count = 0
    stop_count = 0

    def __init__(self, model, patience, stop_patience, threshold, factor, dwell, model_name, freeze, batches, initial_epoch, epochs, ask_epoch):
        super(LRA, self).__init__()
        self.epochs = epochs
        self.ask_epoch = ask_epoch
        self.model = model
        self.patience = patience
        self.stop_patience = stop_patience
        self.threshold = threshold
        self.factor = factor
        self.dwell = dwell
        # get the initiallearning rate and save it in self.lr
        self.lr = float(tf.keras.backend.get_value(model.optimizer.lr))
        self.highest_tracc = 0.0
        self.lowest_vloss = np.inf
        # self.count=0
        # self.stop_count=0
        self.initial_epoch = initial_epoch
        self.batches = batches
        # self.epochs=epochs
        best_weights = self.model.get_weights()
        msg = ' '
        if freeze == True:
            msgs = f'Starting training using base model {model_name} with weights frozen to imagenet weights initializing LRA callback'
        else:
            msgs = f'Starting training using base model {model_name} training all layers '
        print_in_color(msgs, (244, 252, 3), (55, 65, 80))

    def on_train_begin(self, logs=None):
        msg = '{0:^8s}{1:^10s}{2:^9s}{3:^9s}{4:^9s}{5:^9s}{6:^9s}{7:^10s}{8:^8s}'.format('Epoch', 'Loss', 'Accuracy',
                                                                                         'V_loss', 'V_acc', 'LR', 'Next LR', 'Monitor', 'Duration')
        print_in_color(msg, (244, 252, 3), (55, 65, 80))

    def on_train_end(self, logs=None):
        model.set_weights(LRA.best_weights)
        msg = 'Training is completed - model is set with weights for the epoch with the lowest loss'
        print_in_color(msg, (0, 255, 0), (55, 65, 80))

    def on_train_batch_end(self, batch, logs=None):
        acc = logs.get('accuracy') * 100
        loss = logs.get('loss')
        msg = '{0:20s}processing batch {1:4s} of {2:5s} accuracy= {3:8.3f}  loss: {4:8.5f}'.format(
            ' ', str(batch), str(self.batches), acc, loss)
        print(msg, '\r', end='')

    def on_epoch_begin(self, epoch, logs=None):
        self.now = time.time()

    def on_epoch_end(self, epoch, logs=None):
        later = time.time()
        duration = later-self.now
        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        current_lr = lr
        v_loss = logs.get('val_loss')
        acc = logs.get('accuracy')
        v_acc = logs.get('val_accuracy')
        loss = logs.get('loss')
        #print ( '\n',v_loss, self.lowest_vloss, acc, self.highest_tracc)
        if acc < self.threshold:
            monitor = 'accuracy'
            if acc > self.highest_tracc:
                self.highest_tracc = acc
                LRA.best_weights = self.model.get_weights()
                self.count = 0
                self.stop_count = 0
                if v_loss < self.lowest_vloss:
                    self.lowest_vloss = v_loss
                color = (0, 255, 0)
                self.lr = lr
            else:
                if self.count >= self.patience - 1:
                    color = (245, 170, 66)
                    self.lr = lr * self.factor
                    tf.keras.backend.set_value(
                        self.model.optimizer.lr, self.lr)
                    self.count = 0
                    self.stop_count = self.stop_count + 1
                    if self.dwell:
                        self.model.set_weights(LRA.best_weights)
                    else:
                        if v_loss < self.lowest_vloss:
                            self.lowest_vloss = v_loss
                else:
                    self.count = self.count + 1
        else:
            monitor = 'val_loss'
            if v_loss < self.lowest_vloss:
                self.lowest_vloss = v_loss
                LRA.best_weights = self.model.get_weights()
                self.count = 0
                self.stop_count = 0
                color = (0, 255, 0)
                self.lr = lr
            else:
                if self.count >= self.patience-1:
                    color = (245, 170, 66)
                    self.lr = self.lr * self.factor
                    self.stop_count = self.stop_count + 1
                    self.count = 0
                    tf.keras.backend.set_value(
                        self.model.optimizer.lr, self.lr)
                    if self.dwell:
                        self.model.set_weights(LRA.best_weights)
                else:
                    self.count = self.count + 1
                if acc > self.highest_tracc:
                    self.highest_tracc = acc
        msg = f'{str(epoch+1):^3s}/{str(self.epochs):4s} {loss:^9.3f}{acc*100:^9.3f}{v_loss:^9.5f}{v_acc*100:^9.3f}{current_lr:^9.5f}{self.lr:^9.5f}{monitor:^11s}{duration:^8.2f}'
        print_in_color(msg, color, (55, 65, 80))
        if self.stop_count > self.stop_patience - 1:
            msg = f'Training has been halted at epoch {epoch + 1} after {self.stop_patience} adjustments of learning rate with no improvement'
            print_in_color(msg, (0, 255, 255), (55, 65, 80))
            self.model.stop_training = True  # stop training
        else:
            if self.ask_epoch != None:
                if epoch + 1 >= self.ask_epoch:
                    msg = 'Enter H to halt training or an integer for number of epochs to run then ask again'
                    print_in_color(msg, (0, 255, 255), (55, 65, 80))
                    ans = input('')
                    if ans == 'H' or ans == 'h':
                        msg = f'Training has been halted at epoch {epoch + 1} due to user input'
                        print_in_color(msg, (0, 255, 255), (55, 65, 80))
                        self.model.stop_training = True
                    else:
                        ans = int(ans)
                        self.ask_epoch += ans


fpath = r'colored_images/colored_images/Moderate/396_left.png'
img = plt.imread(fpath)
print(img.shape)
imshow(img)

sdir = r'colored_images/colored_images'
classlist = os.listdir(sdir)
filepaths = []
labels = []

for klass in classlist:
    classpath = os.path.join(sdir, klass)
    if os.path.isdir(classpath):
        flist = os.listdir(classpath)
        for f in flist:
            fpath = os.path.join(classpath, f)
            filepaths.append(fpath)
            labels.append(klass)

Fseries = pd.Series(filepaths, name='filepaths')
Lseries = pd.Series(labels, name='labels')
df = pd.concat([Fseries, Lseries], axis=1)
print(df.head())
print('df length: ', len(df))
print(df['labels'].value_counts())

sample_list = []
max_size = 25000
groups = df.groupby('labels')

for label in df['labels'].unique():
    group = groups.get_group(label)
    sample_count = len(group)
    if sample_count > max_size:
        samples = group.sample(max_size, replace=False, weights=None,
                               random_state=123, axis=0).reset_index(drop=True)
    else:
        samples = group.sample(frac=1.0, replace=False,
                               random_state=123, axis=0).reset_index(drop=True)
    sample_list.append(samples)

df = pd.concat(sample_list, axis=0).reset_index(drop=True)
print(len(df))
print(df['labels'].value_counts())

working_dir = r'./'
aug_dir = os.path.join(working_dir, 'aug')
if os.path.isdir(aug_dir):
    shutil.rmtree(aug_dir)
os.mkdir(aug_dir)
for label in df['labels'].unique():
    dir_path = os.path.join(aug_dir, label)
    os.mkdir(dir_path)
print(os.listdir(aug_dir))

target = 25000
gen = ImageDataGenerator(horizontal_flip=True,  rotation_range=20, width_shift_range=.2,
                         height_shift_range=.2, zoom_range=.2)
groups = df.groupby('labels')
for label in df['labels'].unique():
    group = groups.get_group(label)
    sample_count = len(group)
    if sample_count < target:
        aug_img_count = 0
        delta = target-sample_count
        target_dir = os.path.join(aug_dir, label)
        aug_gen = gen.flow_from_dataframe(group,  x_col='filepaths', y_col=None, target_size=(224, 224), class_mode=None,
                                          batch_size=1, shuffle=False, save_to_dir=target_dir, save_prefix='aug-',
                                          save_format='jpg')
        while aug_img_count < delta:
            images = next(aug_gen)
            aug_img_count += len(images)

aug = r'./aug'
auglist = os.listdir(aug)
print(auglist)
for klass in auglist:
    classpath = os.path.join(aug, klass)
    flist = os.listdir(classpath)
    print('klass: ', klass, '  file count: ', len(flist))

plt.figure(figsize=(20, 20))
for i in range(25):
    image = next(aug_gen)/255
    image = np.squeeze(image, axis=0)
    plt.subplot(5, 5, i+1)
    plt.imshow(image)
plt.show()

aug_fpaths = []
aug_labels = []

classlist = os.listdir(aug_dir)

for klass in classlist:
    classpath = os.path.join(aug_dir, klass)
    flist = os.listdir(classpath)
    for f in flist:
        fpath = os.path.join(classpath, f)
        aug_fpaths.append(fpath)
        aug_labels.append(klass)

Fseries = pd.Series(aug_fpaths, name='filepaths')
Lseries = pd.Series(aug_labels, name='labels')
aug_df = pd.concat([Fseries, Lseries], axis=1)
ndf = pd.concat([df, aug_df], axis=0).reset_index(drop=True)

print(df['labels'].value_counts())
print(aug_df['labels'].value_counts())
print(ndf['labels'].value_counts())

train_split = .8
valid_split = .1
dummy_split = valid_split/(1-train_split)
train_df, dummy_df = train_test_split(
    ndf, train_size=train_split, shuffle=True, random_state=123)
valid_df, test_df = train_test_split(
    dummy_df, train_size=dummy_split, shuffle=True, random_state=123)
print('train_df length: ', len(train_df), '  test_df length: ',
      len(test_df), '  valid_df length: ', len(valid_df))

height = 224
width = 224
channels = 3
batch_size = 40
img_shape = (height, width, channels)
img_size = (height, width)
length = len(test_df)
test_batch_size = sorted([int(length/n) for n in range(1, length+1)
                          if length % n == 0 and length/n <= 80], reverse=True)[0]
test_steps = int(length/test_batch_size)
print('test batch size: ', test_batch_size, '  test steps: ', test_steps)


def scalar(img):
    # img=img/127.5-1
    return img


trgen = ImageDataGenerator(preprocessing_function=scalar, horizontal_flip=True)
tvgen = ImageDataGenerator(preprocessing_function=scalar)
train_gen = trgen.flow_from_dataframe(train_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical',
                                      color_mode='rgb', shuffle=True, batch_size=batch_size)
test_gen = tvgen.flow_from_dataframe(test_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical',
                                     color_mode='rgb', shuffle=False, batch_size=test_batch_size)
valid_gen = tvgen.flow_from_dataframe(valid_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical',
                                      color_mode='rgb', shuffle=True, batch_size=batch_size)
classes = list(train_gen.class_indices.keys())
class_count = len(classes)
train_steps = int(len(train_gen.labels)/batch_size)


def show_image_samples(gen):
    t_dict = gen.class_indices
    classes = list(t_dict.keys())
    images, labels = next(gen)
    plt.figure(figsize=(20, 20))
    length = len(labels)
    if length < 25:
        r = length
    else:
        r = 25
    for i in range(r):
        plt.subplot(5, 5, i + 1)
        image = images[i]/255
        plt.imshow(image)
        index = np.argmax(labels[i])
        class_name = classes[index]
        plt.title(class_name, color='blue', fontsize=16)
        plt.axis('off')
    plt.show()


show_image_samples(train_gen)


def print_in_color(txt_msg, fore_tupple, back_tupple,):
    rf, gf, bf = fore_tupple
    rb, gb, bb = back_tupple
    msg = '{0}' + txt_msg
    mat = '\33[38;2;' + str(rf) + ';' + str(gf) + ';' + str(bf) + \
        ';48;2;' + str(rb) + ';' + str(gb) + ';' + str(bb) + 'm'
    print(msg .format(mat), flush=True)
    print('\33[0m', flush=True)
    return


model = tf.keras.Sequential([
    efn.EfficientNetB0(
        input_shape=(224, 224, 3),
        weights='imagenet',
        include_top=False
    ),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

model.compile(Adamax(lr=0.001), loss='categorical_crossentropy',
              metrics=['accuracy', 'AUC'])

epochs = 100
patience = 1
stop_patience = 3
threshold = .9
factor = .5
dwell = True
freeze = False
ask_epoch = 10
batches = train_steps

callbacks = [LRA(model=model, patience=patience, stop_patience=stop_patience, threshold=threshold,
                 factor=factor, dwell=dwell, model_name=model_name, freeze=freeze, batches=batches, initial_epoch=0, epochs=epochs, ask_epoch=ask_epoch)]

history = model.fit(x=train_gen,  epochs=epochs, verbose=0, callbacks=callbacks,  validation_data=valid_gen,
                    validation_steps=None,  shuffle=False,  initial_epoch=0)


def tr_plot(tr_data, start_epoch):
    tacc = tr_data.history['accuracy']
    tloss = tr_data.history['loss']
    vacc = tr_data.history['val_accuracy']
    vloss = tr_data.history['val_loss']
    Epoch_count = len(tacc) + start_epoch
    Epochs = []
    for i in range(start_epoch, Epoch_count):
        Epochs.append(i+1)
    index_loss = np.argmin(vloss)
    val_lowest = vloss[index_loss]
    index_acc = np.argmax(vacc)
    acc_highest = vacc[index_acc]
    plt.style.use('fivethirtyeight')
    sc_label = 'best epoch= ' + str(index_loss+1 + start_epoch)
    vc_label = 'best epoch= ' + str(index_acc + 1 + start_epoch)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))
    axes[0].plot(Epochs, tloss, 'r', label='Training loss')
    axes[0].plot(Epochs, vloss, 'g', label='Validation loss')
    axes[0].scatter(index_loss+1 + start_epoch, val_lowest,
                    s=150, c='blue', label=sc_label)
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[1].plot(Epochs, tacc, 'r', label='Training Accuracy')
    axes[1].plot(Epochs, vacc, 'g', label='Validation Accuracy')
    axes[1].scatter(index_acc+1 + start_epoch, acc_highest,
                    s=150, c='blue', label=vc_label)
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    plt.tight_layout
    plt.show()


def print_info(test_gen, preds, print_code, save_dir, subject):
    class_dict = test_gen.class_indices
    labels = test_gen.labels
    file_names = test_gen.filenames
    error_list = []
    true_class = []
    pred_class = []
    prob_list = []
    new_dict = {}
    error_indices = []
    y_pred = []
    for key, value in class_dict.items():
        new_dict[value] = key
    classes = list(new_dict.values())
    dict_as_text = str(new_dict)
    dict_name = subject + '-' + str(len(classes)) + '.txt'
    dict_path = os.path.join(save_dir, dict_name)
    with open(dict_path, 'w') as x_file:
        x_file.write(dict_as_text)
    errors = 0
    for i, p in enumerate(preds):
        pred_index = np.argmax(p)
        true_index = labels[i]
        if pred_index != true_index:
            error_list.append(file_names[i])
            true_class.append(new_dict[true_index])
            pred_class.append(new_dict[pred_index])
            prob_list.append(p[pred_index])
            error_indices.append(true_index)
            errors = errors + 1
        y_pred.append(pred_index)
    if print_code != 0:
        if errors > 0:
            if print_code > errors:
                r = errors
            else:
                r = print_code
            msg = '{0:^28s}{1:^28s}{2:^28s}{3:^16s}'.format(
                'Filename', 'Predicted Class', 'True Class', 'Probability')
            print_in_color(msg, (0, 255, 0), (55, 65, 80))
            for i in range(r):
                split1 = os.path.split(error_list[i])
                split2 = os.path.split(split1[0])
                fname = split2[1] + '/' + split1[1]
                msg = '{0:^28s}{1:^28s}{2:^28s}{3:4s}{4:^6.4f}'.format(
                    fname, pred_class[i], true_class[i], ' ', prob_list[i])
                print_in_color(msg, (255, 255, 255), (55, 65, 60))
                #print(error_list[i]  , pred_class[i], true_class[i], prob_list[i])
        else:
            msg = '100%\t accuracy. There are no errors.'
            print_in_color(msg, (0, 255, 0), (55, 65, 80))
    if errors > 0:
        plot_bar = []
        plot_class = []
        for key, value in new_dict.items():
            count = error_indices.count(key)
            if count != 0:
                plot_bar.append(count)
                plot_class.append(value)
        fig = plt.figure()
        fig.set_figheight(len(plot_class)/3)
        fig.set_figwidth(10)
        plt.style.use('fivethirtyeight')
        for i in range(0, len(plot_class)):
            c = plot_class[i]
            x = plot_bar[i]
            plt.barh(c, x, )
            plt.title('Errors by Class on Test Set')
    y_true = np.array(labels)
    y_pred = np.array(y_pred)
    if len(classes) <= 30:
        cm = confusion_matrix(y_true, y_pred)
        length = len(classes)
        if length < 8:
            fig_width = 8
            fig_height = 8
        else:
            fig_width = int(length * .5)
            fig_height = int(length * .5)
        plt.figure(figsize=(fig_width, fig_height))
        sns.heatmap(cm, annot=True, vmin=0, fmt='g', cmap='Blues', cbar=False)
        plt.xticks(np.arange(length)+.5, classes, rotation=90)
        plt.yticks(np.arange(length)+.5, classes, rotation=0)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()
    clr = classification_report(y_true, y_pred, target_names=classes)
    print("Classification Report:\n----------------------\n", clr)


tr_plot(history, 0)
save_dir = r'./'
subject = 'plants'
acc = model.evaluate(test_gen, batch_size=test_batch_size,
                     verbose=1, steps=test_steps, return_dict=False)[1]*100
msg = f'Accuracy on the test set is {acc:5.2f} %'
print_in_color(msg, (0, 255, 0), (55, 65, 80))
save_id = str(model_name + '-' + subject + '-' +
              str(acc)[:str(acc).rfind('.')+3] + '.h5')
save_loc = os.path.join(save_dir, save_id)
model.save(save_loc)

print_code = 0
preds = model.predict(test_gen)
print_info(test_gen, preds, print_code, save_dir, subject)
