import numpy as np
import tensorflow as tf
from tools import *
import sys
import os
import time
import math

class S2VT:
    def __init__(self,sess,Infer=False):
        self.n_steps = 80
        self.hidden_dim = 1000
        self.emb_dim = 500
        self.frame_dim = 4096
        if Infer==True:
            self.batch_size=1
        else:
            self.batch_size = 10
        self.learning_rate=0.0001
        self.dataNum=10000
        self.epochNum=200
        self.vocab_size = len(word2id)
        self.bias_init_vector = get_bias_vector()
        self.checkpointDir = "./checkpoint"
        self.sess=sess
        self.modelName="S2VT"
        
    def buildModel(self):
        """This function creates weight matrices that transform:
            * frames to caption dimension
            * hidden state to vocabulary dimension
            * creates word embedding matrix """

        print("Network config: \nN_Steps: {}\nHidden_dim:{}\nEnb_dim:{}\nFrame_dim:{}\nBatch_size:{}\nVocab_size:{}\n".format(self.n_steps,\
                    self.hidden_dim,self.emb_dim,self.frame_dim,self.batch_size,self.vocab_size))

        n_steps=self.n_steps
        hidden_dim=self.hidden_dim
        emb_dim=self.emb_dim
        frame_dim=self.frame_dim
        batch_size=self.batch_size
        vocab_size=self.vocab_size
        #Create placeholders for holding a batch of videos, captions and caption masks
        video = tf.placeholder(tf.float32,shape=[batch_size,n_steps,frame_dim],name='Input_Video')
        # 10 videos , 80 time steps , 4096 dims feature

        caption = tf.placeholder(tf.int32,shape=[batch_size,n_steps],name='GT_Caption')
        # Ground Truth

        caption_mask = tf.placeholder(tf.float32,shape=[batch_size,n_steps],name='Caption_Mask')
        # Mask

        dropout_prob = tf.placeholder(tf.float32,name='Dropout_Keep_Probability')
        # dropout rate

        with tf.variable_scope('Im2Cap') as scope:
            W_im2cap = tf.get_variable(name='W_im2cap',shape=[frame_dim,emb_dim],\
                initializer=tf.random_uniform_initializer(minval=-0.08,maxval=0.08))

            b_im2cap = tf.get_variable(name='b_im2cap',shape=[emb_dim],\
                initializer=tf.constant_initializer(0.0))
        # 4096维的特征向量要被降维成500维，feature_vec * W_im2cap + b_im2cap    
        
        with tf.variable_scope('Hid2Vocab') as scope:
            W_H2vocab = tf.get_variable(name='W_H2vocab',shape=[hidden_dim,vocab_size],\
                initializer=tf.random_uniform_initializer(minval=-0.08,maxval=0.08))

            b_H2vocab = tf.Variable(name='b_H2vocab',initial_value=self.bias_init_vector.astype(np.float32))
        # LSTM最后输出的1000维的向量，要被处理成vocal_size维的词向量
        
        with tf.variable_scope('Word_Vectors') as scope:
            word_emb = tf.get_variable(name='Word_embedding',shape=[vocab_size,emb_dim],
                initializer=tf.random_uniform_initializer(minval=-0.08,maxval=0.08))
        # id -> vector 矩阵
        print("Created weights")

        #Build two LSTMs, one for processing the video and another for generating the caption
        with tf.variable_scope('LSTM_Video',reuse=None) as scope:
            lstm_vid = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim)
            lstm_vid = tf.nn.rnn_cell.DropoutWrapper(lstm_vid,output_keep_prob=dropout_prob)

        with tf.variable_scope('LSTM_Caption',reuse=None) as scope:
            lstm_cap = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim)
            lstm_cap = tf.nn.rnn_cell.DropoutWrapper(lstm_cap,output_keep_prob=dropout_prob)

        #Prepare input for lstm_video
        video_rshp = tf.reshape(video,[-1,frame_dim])
        video_rshp = tf.nn.dropout(video_rshp,keep_prob=dropout_prob)
        video_emb = tf.nn.xw_plus_b(video_rshp,W_im2cap,b_im2cap)
        video_emb = tf.reshape(video_emb,[batch_size,n_steps,emb_dim])
        # 500维的视频向量
        padding = tf.zeros([batch_size,n_steps-1,emb_dim])
        video_input = tf.concat([video_emb,padding],1)
        # n_step次视频输入 + n_step-1次zero padding
        print("Video_input: {}".format(video_input.get_shape()))
        
        #Run lstm_vid for 2*n_steps-1 timesteps
        with tf.variable_scope('LSTM_Video') as scope:
            out_vid,state_vid = tf.nn.dynamic_rnn(lstm_vid,video_input,dtype=tf.float32)
        print("Video_output: {}".format(out_vid.get_shape()))


        #Prepare input for lstm_cap
        padding = tf.zeros([batch_size,n_steps,emb_dim]) #n步的词向量padding
        caption_vectors = tf.nn.embedding_lookup(word_emb,caption[:,0:n_steps-1]) 
        #根据GT的id来索引词向量
        caption_vectors = tf.nn.dropout(caption_vectors,keep_prob=dropout_prob)
        caption_2n = tf.concat([padding,caption_vectors],1) #2n-1次词向量输入
        caption_input = tf.concat([caption_2n,out_vid],2) #词向量输入和第一层LSTM输出的联合
        print("Caption_input: {}".format(caption_input.get_shape()))
        #Run lstm_cap for 2*n_steps-1 timesteps
        with tf.variable_scope('LSTM_Caption') as scope:
            out_cap,state_cap = tf.nn.dynamic_rnn(lstm_cap,caption_input,dtype=tf.float32)
        print("Caption_output: {}".format(out_cap.get_shape()))


        #Compute masked loss
        output_captions = out_cap[:,n_steps:,:] 
        output_logits = tf.reshape(output_captions,[-1,hidden_dim])
        output_logits = tf.nn.dropout(output_logits,keep_prob=dropout_prob)
        output_logits = tf.nn.xw_plus_b(output_logits,W_H2vocab,b_H2vocab) #生成一个维度为vocabulary_size的向量，作为每个词的概率
        
        output_labels = tf.reshape(caption[:,1:],[-1]) #label : 过滤掉GT的第一个<BOS>,因为本来就不会生成。。
        caption_mask_out = tf.reshape(caption_mask[:,1:],[-1])# mask 同理
        #全部都作一维展开

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output_logits,labels=output_labels)
        masked_loss = loss*caption_mask_out
        loss = tf.reduce_sum(masked_loss)/tf.reduce_sum(caption_mask_out)
        
        return video,caption,caption_mask,output_logits,loss,dropout_prob


    def train(self):

        video,caption,caption_mask,output_logits,loss,dropout_prob = self.buildModel()
        optim = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(loss)
        nIter = int(self.dataNum/self.batch_size)

        self.saver = tf.train.Saver(max_to_keep=25)
        
        startEpoch = 0
        counter = 0

        hasCheckpoint , checkpointCounter = self.loadModel(self.checkpointDir)
        if hasCheckpoint:
            startEpoch=checkpointCounter
            counter=checkpointCounter
            print("模型加载成功")
        else:
            self.sess.run(tf.initialize_all_variables())
            print("没有找到检查点，模型从初始化状态开始训练")

        

        print("训练开始！～～～～～～～～～～～～～～～～～～～～～～～～")
        timeStart = time.time()
        for epoch in range(startEpoch,self.epochNum):
            for bid in range(nIter):
                vids,caps,caps_mask,vid_urls,needPadding = fetch_data_batch(batch_size=self.batch_size)
                #print(vids.shape,caps.shape,caps_mask.shape)
                #input("pause")
                try:
                    _,curr_loss,o_l = self.sess.run([optim,loss,output_logits],feed_dict=\
                        {video:vids,caption:caps,caption_mask:caps_mask,dropout_prob:0.5})
                except:
                    print(vids.shape,caps.shape,caps_mask.shape,vid_urls.shape)
                    print(vid_urls)
                    for i in range(self.batch_size):
                        print(vids[i].shape)
                    input("pause")
                    
                   # input("pause")
                if math.isnan(curr_loss):
                    print("NaN!!!")
                    print("Padding : ",needPadding)
                    print(caps)
                    print(caps_mask)
                    print(vids.shape)
                    MIN=99999999
                    MAX=-99999999
                    for tt in range(self.batch_size):
                        for ttt in range(self.n_steps):
                            for tttt in range(self.frame_dim):
                                #print(tt,ttt,tttt)
                                tempN=vids[tt][ttt][tttt]
                                if math.isnan(tempN):
                                    print(tempN)
                                else :
                                    MIN=min(MIN,tempN)
                                    MAX=max(MAX,tempN)
                    print("MIN : ",MIN)
                    print("MAX : ",MAX)
                    print(vid_urls)
                    input("PAUSE")



                print("Epoch: [%4d/%4d] [%4d/%4d] time: %4.4f seconds, loss: %.8f" \
                    % (epoch, self.epochNum, bid, nIter, time.time() - timeStart, curr_loss), end='\r\n') 
            counter +=1
            print()
            self.saveModel(self.checkpointDir,counter)
    

    def test(self):
        video,caption,caption_mask,output_logits,loss,dropout_prob = self.buildModel()
        self.saver = tf.train.Saver(max_to_keep=25)
        hasCheckpoint , checkpointCounter = self.loadModel(self.checkpointDir)
        if hasCheckpoint:
            startEpoch=checkpointCounter
            counter=checkpointCounter
            print("模型加载成功")
        else:
            #self.sess.run(tf.initialize_all_variables())
            print("没有找到模型，无法推理")
            exit(0)

        while(True):
            vid,caption_GT,_,video_urls,_ = fetch_data_batch(1)
            caps,caps_mask = convert_for_test(['<BOS>'],80)
            for i in range(self.n_steps):
                #print(caps)
                #print(caps_mask)
                #input("pause")
                o_l = self.sess.run(output_logits,feed_dict={video:vid,
                                                        caption:caps,
                                                        caption_mask:caps_mask,
                                                        dropout_prob:1.0})
                out_logits = o_l.reshape([self.batch_size,self.n_steps-1,self.vocab_size])
                output_captions = np.argmax(out_logits,2)
                caps[0][i+1] = output_captions[0][i]
                print_in_english(output_captions)
                if id2word[output_captions[0][i]] == '<EOS>':
                    break

            print('video link : ',video_urls[0])
            print('............................\nGT Caption:\n')
            print_in_english(caption_GT)
            play_video = input('Should I play the video? y/n')
            if play_video.lower() == 'y':
                    #playVideo(video_urls)
                pass
            test_again = input('Want another test run? y/n')
            if test_again.lower() == 'n':
                break


    def saveModel(self,checkpointDir,step):           #保存模型
        checkpointDir = os.path.join(checkpointDir,self.modelName)
        if not os.path.exists(checkpointDir):
            os.makedirs(checkpointDir)
        print("新的模型将会被保存 : ",checkpointDir)
        self.saver.save(self.sess,os.path.join(checkpointDir,self.modelName+".model"),global_step=step)
        

    def loadModel(self,checkpointDir):               #加载模型
        checkpointDir = os.path.join(checkpointDir,self.modelName)
        print("模型将会被加载 : ",checkpointDir)
        ckpt = tf.train.get_checkpoint_state(checkpointDir)  #从模型保存的目录下拉取模型信息
        if ckpt and ckpt.model_checkpoint_path:
            modelName = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess,os.path.join(checkpointDir,modelName))
            print("MODEL NAME : ",modelName)
            st = modelName.index("-")
            counter = int(modelName[st+1:])
            print("模型加载成功 : ",modelName)
            return True,counter
        else :
            print("没有找到检查点")
            return False,0
    

