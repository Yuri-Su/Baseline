from keras import layers
import tensorflow as tf

def BN():
    return layers.BatchNormalization2()

def SyncBN():
    return layers.SyncBatchNormalization()

def LN():
    return layers.LayerNormalization()