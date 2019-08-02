import numpy as np
import json
import glob
import cv2
import imageio

jFile = open("./data/videodatainfo_2017_ustc.json")
load_dict = json.load(jFile)
sents = load_dict["sentences"]
videoList = {}
for t in sents:
    x=t["video_id"]
    if x not in videoList:
        videoList[x]=[]
    videoList[x].append(t["caption"])

videoName = list(videoList.keys())


def build_vocab(word_count_thresh):
    """Function to create vocabulary based on word count threshold.
        Input:
                word_count_thresh: Threshold to choose words to include to the vocabulary
        Output:
                vocabulary: Set of words in the vocabulary
    """

    unk_required = False  #是否需要Unknown
    word_counts = {}  

    for temp in sents:
        caption=temp["caption"]
        caption = '<BOS> ' + caption + ' <EOS>'
        #if(len(caption.split(' '))>=75):
        #    print(len(caption.split(' ')))
        for word in caption.split(' '):
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1
    
    newList = {}

    for word in word_counts.keys():
        if word_counts[word] < word_count_thresh:
            #word_counts.pop(word)
            unk_required = True
        else :
            newList[word]=word_counts[word]

    word_counts=newList

    return word_counts,unk_required

def word_to_word_ids(word_counts,unk_required):
    """Function to map individual words to their id's.
        Input:
                word_counts: Dictionary with words mapped to their counts
        Output:
                word_to_id: Dictionary with words mapped to their id's. """
    count = 0
    word_to_id = {}
    id_to_word = {}
    if unk_required:
        count += 1
        word_to_id['<UNK>'] = count
        id_to_word[count] = '<UNK>'
    for word in word_counts.keys():
        count += 1
        word_to_id[word] = count
        id_to_word[count] = word
    return word_to_id,id_to_word


word_counts,unk_required = build_vocab(0)
n_lstm_steps = 80
word2id,id2word = word_to_word_ids(word_counts,unk_required)

def convert_caption(caption,max_caption_length):
    """
        Function to map each word in a caption to it's respective id and to retrieve caption masks
        Input:
                caption: Caption to convert to word_to_word_ids
                word_to_id: Dictionary mapping words to their respective id's
                max_caption_length: Maximum number of words allowed in a caption
        Output:
                caps: Captions with words mapped to word id's
                cap_masks: Caption masks with 1's at positions of words and 0's at pad locations
    """
    word_to_id = word2id
    caps,cap_masks = [],[]

    if type(caption) == 'str':
        caption = [caption] # if single caption, make it a list of captions of length one

    for cap in caption:
        cap='<BOS> '+cap+' <EOS>'
        nWords = cap.count(' ') + 1
        cap = cap + ' <EOS>'*(max_caption_length-nWords)
        cap_masks.append([1.0]*nWords + [0.0]*(max_caption_length-nWords))
        curr_cap = []
        for word in cap.split(' '):
            if word in word_to_id:
                curr_cap.append(word_to_id[word]) # word is present in chosen vocabulary
            else:
                curr_cap.append(word_to_id['<UNK>']) # word not present in chosen vocabulary
        caps.append(curr_cap)

    return np.array(caps),np.array(cap_masks) ## id序列 + mask

def convert_for_test(caption,max_caption_length):
    """
        Function to map each word in a caption to it's respective id and to retrieve caption masks
        Input:
                caption: Caption to convert to word_to_word_ids
                word_to_id: Dictionary mapping words to their respective id's
                max_caption_length: Maximum number of words allowed in a caption
        Output:
                caps: Captions with words mapped to word id's
                cap_masks: Caption masks with 1's at positions of words and 0's at pad locations
    """
    word_to_id = word2id
    caps,cap_masks = [],[]

    if type(caption) == 'str':
        caption = [caption] # if single caption, make it a list of captions of length one

    for cap in caption:
        nWords = cap.count(' ') + 1
        cap = cap + ' <EOS>'*(max_caption_length-nWords)
        cap_masks.append([1.0]*nWords + [0.0]*(max_caption_length-nWords))
        curr_cap = []
        for word in cap.split(' '):
            if word in word_to_id:
                curr_cap.append(word_to_id[word]) # word is present in chosen vocabulary
            else:
                curr_cap.append(word_to_id['<UNK>']) # word not present in chosen vocabulary
        caps.append(curr_cap)

    return np.array(caps),np.array(cap_masks) ## id序列 + mask



def get_bias_vector():
    """Function to return the initialization for the bias vector
       for mapping from hidden_dim to vocab_size.
       Borrowed from neuraltalk by Andrej Karpathy"""
    bias_init_vector = np.array([1.0*word_counts[id2word[i]] for i in id2word])
    bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector) # log p
    bias_init_vector -= np.max(bias_init_vector) # log p - max(log p)
    return bias_init_vector

def fetch_data_batch(batch_size):
    """
        Function to fetch a batch of video features, captions and caption masks
        Input:
                batch_size: Size of batch to load
        Output:
                curr_vids: Features of the randomly selected batch of video_files
                curr_caps: Ground truth (padded) captions for the selected videos
                curr_masks: Mask for the pad locations in curr_caps
    """
    
    curr_batch_vids = np.random.choice(videoName,batch_size) #video object
    video_urls = np.array(["./rgb_feats/"+vid+".mp4.npy" for vid in curr_batch_vids]) #
    video_loc =np.array(["../train_video/"+vid+".mp4" for vid in curr_batch_vids]) #
    curr_vids = np.array([np.load(vid) for vid in video_urls])
    
    needPadding=False

    for i in range(batch_size):
        temp=curr_vids[i]
        frame,_=temp.shape
        if(frame<80):
            needPadding=True
            diff=80-frame
            padding = np.zeros([diff,4096])
            curr_vids[i]=np.concatenate((temp,padding))  
    
    captionList = [videoList[vidName] for vidName in curr_batch_vids]
    curr_captions = np.array([np.random.choice(temp,1)[0] for temp in captionList])
    # caption , video_feature
    curr_caps,curr_masks = convert_caption(curr_captions,n_lstm_steps)
    
    if needPadding==True:
        newVidList = []
        for i in range(batch_size):
            newVidList.append(curr_vids[i])
            #print(curr_vids[i].shape)
        curr_vids=np.array(newVidList)

    #curr_vids.reshape((batch_size,80,4096))
    return curr_vids,curr_caps,curr_masks,video_loc,needPadding

def print_in_english(caption_idx):
    """Function to take a list of captions with words mapped to ids and
        print the captions after mapping word indices back to words."""
    captions_english = [[id2word[word] for word in caption] for caption in caption_idx]
    for i,caption in enumerate(captions_english):
        if '<EOS>' in caption:
       	    caption = caption[0:caption.index('<EOS>')]
        print(caption)
        print('..................................................')

def playVideo(video_urls):
    video = imageio.get_reader(video_urls[0],'ffmpeg')
    for frame in video:
        fr = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        cv2.imshow('frame',fr)
        if cv2.waitKey(40) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()