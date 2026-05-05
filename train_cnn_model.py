"""
訓練 CNN 模型用於蠟燭圖案分類
基於 FinancialVision 的程式碼
"""
import sys
import os
from pathlib import Path

# 加入 utils 路徑
sys.path.insert(0, str(Path(__file__).parent / "FinancialVision-master" / "Encoding candlesticks as images for patterns classification using convolutional neural networks" / "utils"))

from sklearn.metrics import confusion_matrix
import numpy as np
import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Activation

# 自訂工具
import util_process as pro


def get_model(params):
    model = Sequential()

    # Conv1
    model.add(Conv2D(16, (2, 2), input_shape=(10, 10, 4), padding='same', strides=(1, 1)))
    model.add(Activation('sigmoid'))

    # Conv2
    model.add(Conv2D(16, (2, 2), padding='same', strides=(1, 1)))
    model.add(Activation('sigmoid'))

    # FC
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))

    model.add(Dense(params['classes']))
    model.add(Activation('softmax'))
    model.summary()

    return model


def train_model(params, data):
    model = get_model(params)
    model.compile(loss='categorical_crossentropy', optimizer=params['optimizer'], metrics=['accuracy'])
    hist = model.fit(x=data['train_gaf'], y=data['train_label_arr'],
                     validation_data=(data['val_gaf'], data['val_label_arr']),
                     batch_size=params['batch_size'], epochs=params['epochs'], verbose=2)
    
    return (model, hist)


def print_result(data, model):
    # 取得訓練和測試的預測標籤
    train_pred = np.argmax(model.predict(data['train_gaf'], verbose=0), axis=1)
    test_pred = np.argmax(model.predict(data['test_gaf'], verbose=0), axis=1)
    # 取得訓練和測試的真實標籤
    train_label = data['train_label'][:, 0]
    test_label = data['test_label'][:, 0]
    # 訓練和測試混淆矩陣
    train_result_cm = confusion_matrix(train_label, train_pred, labels=range(9))
    test_result_cm = confusion_matrix(test_label, test_pred, labels=range(9))

    print("Train Confusion Matrix:")
    print(train_result_cm)
    print("\nTest Confusion Matrix:")
    print(test_result_cm)


if __name__ == "__main__":
    PARAMS = {}
    # 指向 C:\UW_AgenticAI 中的訓練資料
    PARAMS['pkl_name'] = './label8_eurusd_10bar_1500_500_val200_gaf_culr.pkl'
    
    # 模型輸出路徑
    PARAMS['model_name'] = str(Path(__file__).parent / 'cnn_model_10bar.h5')
    
    PARAMS['classes'] = 9
    PARAMS['lr'] = 0.01
    PARAMS['epochs'] = 50
    PARAMS['batch_size'] = 64
    PARAMS['optimizer'] = optimizers.SGD(learning_rate=PARAMS['lr'])

    print(f"Loading training data from: {PARAMS['pkl_name']}")
    print(f"Model will be saved to: {PARAMS['model_name']}")
    
    # 載入資料和模型
    data = pro.load_pkl(PARAMS['pkl_name'])
    print(f"Data loaded successfully!")
    print(f"  Train GAF shape: {data['train_gaf'].shape}")
    print(f"  Val GAF shape: {data['val_gaf'].shape}")
    print(f"  Test GAF shape: {data['test_gaf'].shape}")

    # 訓練 CNN 模型
    print("\nStarting model training...")
    model, hist = train_model(PARAMS, data)
    
    # 保存模型
    model.save(PARAMS['model_name'])
    print(f"\nModel saved to: {PARAMS['model_name']}")
    
    # 訓練和測試結果
    print("\n" + "="*50)
    print("Performance Results:")
    print("="*50)
    print_result(data, model)
