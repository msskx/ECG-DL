#!/usr/bin/env python
# coding: utf-8

import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

from Data.Dataloader import DataLoader

data = DataLoader()
X_train, y_train, X_test, y_test = data.getdata01()



#独热码归一化
y_train=np.argmax(np.array(y_train),axis=1)
y_test=np.argmax(np.array(y_test),axis=1)

x_test=X_test
y_test=y_test


x_train, x_val = X_train[:5226], X_train[5226:]
y_train, y_val = y_train[:5226], y_train[5226:]


# 
def make_pairs(x, y):
    """Creates a tuple containing image pairs with corresponding label.
      创建包含具有相应标签的心电图对的元组
    Arguments:
        x: List containing images, each index in this list corresponds to one image.
        x: 
        y: List containing labels, each label with datatype of `int`.

    Returns:
        Tuple containing two numpy arrays as (pairs_of_samples, labels),
        where pairs_of_samples' shape is (2len(x), 2,n_features_dims) and
        labels are a binary array of shape (2len(x)).
    """

    num_classes = max(y) + 1
    digit_indices = [np.where(y == i)[0] for i in range(num_classes)]

    pairs = []
    labels = []

    for idx1 in range(len(x)):
        # add a matching example
        x1 = x[idx1]
        label1 = y[idx1]
        idx2 = random.choice(digit_indices[label1])
        x2 = x[idx2]

        pairs += [[x1, x2]]
        labels += [0]

        # add a non-matching example
        label2 = random.randint(0, num_classes - 1)
        while label2 == label1:
            label2 = random.randint(0, num_classes - 1)

        idx2 = random.choice(digit_indices[label2])
        x2 = x[idx2]

        pairs += [[x1, x2]]
        labels += [1]

    return np.array(pairs), np.array(labels).astype("float32")


# make train pairs
pairs_train, labels_train = make_pairs(x_train, y_train)

# make validation pairs
pairs_val, labels_val = make_pairs(x_val, y_val)

# make test pairs
pairs_test, labels_test = make_pairs(x_test, y_test)


x_train.shape,x_val.shape,y_train.shape,y_val.shape



pairs_train.shape,labels_train.shape


pairs_test.shape,labels_test.shape



x_train_1 = pairs_train[:, 0]  # x_train_1.shape is (10452, 1000, 12)
x_train_2 = pairs_train[:, 1]



x_train_1.shape


x_val_1 = pairs_val[:, 0]  # x_val_1.shape = (10452, 1000, 12)
x_val_2 = pairs_val[:, 1]


x_test_1 = pairs_test[:, 0]  # x_test_1.shape = (2338, 1000, 12)
x_test_2 = pairs_test[:, 1]




# 定义模型

#有两个输入层，每个输入层都通向自己的网络，该网络产生嵌入。Lambda层使用欧几里得距离将它们合并，并将合并的输出反馈到最终网络。


# # Provided two tensors t1 and t2
# # Euclidean distance = sqrt(sum(square(t1-t2)))
# def euclidean_distance(vects):
#     """
#     求两个向量的欧式距离
#     参数：包含两个长度相同的张量
#     返回：包含向量欧式距离的张量
        
#     """

#     x, y = vects
#     sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
#     return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))


# input = layers.Input((1000, 12, 1))
# x = tf.keras.layers.BatchNormalization()(input)
# x = layers.Conv2D(4, (3, 3), activation="tanh")(x)
# x = layers.AveragePooling2D(pool_size=(2, 2))(x)
# x = layers.Conv2D(16, (3, 3), activation="tanh")(x)
# x = layers.AveragePooling2D(pool_size=(2, 2))(x)
# x = layers.Flatten()(x)

# x = tf.keras.layers.BatchNormalization()(x)
# x = layers.Dense(2, activation="tanh")(x)
# embedding_network = keras.Model(input, x)


# input_1 = layers.Input((1000, 12, 1))
# input_2 = layers.Input((1000, 12, 1))

# # As mentioned above, Siamese Network share weights between
# # tower networks (sister networks). To allow this, we will use
# # same embedding network for both tower networks.
# tower_1 = embedding_network(input_1)
# tower_2 = embedding_network(input_2)

# merge_layer = layers.Lambda(euclidean_distance)([tower_1, tower_2])
# normal_layer = tf.keras.layers.BatchNormalization()(merge_layer)
# output_layer = layers.Dense(1, activation="sigmoid")(normal_layer)
# siamese = keras.Model(inputs=[input_1, input_2], outputs=output_layer)


# Provided two tensors t1 and t2
# Euclidean distance = sqrt(sum(square(t1-t2)))
def euclidean_distance(vects):
    """
    求两个向量的欧式距离
    参数：包含两个长度相同的张量
    返回：包含向量欧式距离的张量
        
    """

    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))


input = layers.Input((1000, 12))
x = tf.keras.layers.BatchNormalization()(input)
x = layers.Conv1D(4, 3, activation="relu")(x)
x = layers.AveragePooling1D(pool_size=2)(x)
x = layers.Conv1D(16, 3, activation="relu")(x)
# x = layers.AveragePooling1D(pool_size=2)(x)
x = layers.Flatten()(x)

x = tf.keras.layers.BatchNormalization()(x)
x = layers.Dense(2, activation="tanh")(x)
embedding_network = keras.Model(input, x)


input_1 = layers.Input((1000, 12))
input_2 = layers.Input((1000, 12))

# As mentioned above, Siamese Network share weights between
# tower networks (sister networks). To allow this, we will use
# same embedding network for both tower networks.
tower_1 = embedding_network(input_1)
tower_2 = embedding_network(input_2)

merge_layer = layers.Lambda(euclidean_distance)([tower_1, tower_2])
normal_layer = tf.keras.layers.BatchNormalization()(merge_layer)
output_layer = layers.Dense(1, activation="sigmoid")(normal_layer)
siamese = keras.Model(inputs=[input_1, input_2], outputs=output_layer)



def loss(margin=1):
    """为'constrastive_loss'提供一个包含变量'margin'的封闭作用域。

    Arguments:
        margin:整数，定义了输入对之间距离的基线
        应归为异类。-(默认为1)。

    Returns:
        'constrastive_loss'函数，附带数据('margin')。
    """

    # Contrastive loss = mean( (1-true_value) * square(prediction) +
    #                         true_value * square( max(margin-prediction, 0) ))
    def contrastive_loss(y_true, y_pred):
        """Calculates the constrastive loss.

        Arguments:
            y_true: List of labels, each label is of type float32.
            y_pred: List of predictions of same length as of y_true,
                    each label is of type float32.

        Returns:
            A tensor containing constrastive loss as floating point value.
        """

        square_pred = tf.math.square(y_pred)
        margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
        return tf.math.reduce_mean(
            (1 - y_true) * square_pred + (y_true) * margin_square
        )

    return contrastive_loss

epochs = 100
batch_size = 128
margin = 2  # Margin for constrastive loss.

siamese.compile(loss=loss(margin=margin), optimizer="RMSprop", metrics=["accuracy"])
siamese.summary()


history = siamese.fit(
    [x_train_1, x_train_2],
    labels_train,
    validation_data=([x_val_1, x_val_2], labels_val),
    batch_size=batch_size,
    epochs=epochs,
)


def plt_metric(history, metric, title, has_valid=True):
    """Plots the given 'metric' from 'history'.

    Arguments:
        history: history attribute of History object returned from Model.fit.
        metric: Metric to plot, a string value present as key in 'history'.
        title: A string to be used as title of plot.
        has_valid: Boolean, true if valid data was passed to Model.fit else false.

    Returns:
        None.
    """
    plt.plot(history[metric])
    if has_valid:
        plt.plot(history["val_" + metric])
        plt.legend(["train", "validation"], loc="upper left")
    plt.title(title)
    plt.ylabel(metric)
    plt.xlabel("epoch")
    plt.show()


#画图
# plt_metric(history=history.history, metric="accuracy", title="Model accuracy")


# plt_metric(history=history.history, metric="loss", title="Constrastive Loss")


results = siamese.evaluate([x_test_1, x_test_2], labels_test)
print("test loss, test acc:", results)


