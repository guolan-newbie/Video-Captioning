from S2VT import S2VT
import tensorflow as tf


with tf.Session() as sess :
    myS2VT = S2VT(sess)
    myS2VT.train()
