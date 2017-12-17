import numpy as np
import os
from model import cap_gen
import pandas as pd


dim_hidden = 256
dim_image = 4096
batch_size = 128

learning_rate = 0.001
momentum = 0.9
n_epochs = 1000

vgg_path = './vgg19.tfmodel'
features_path = './features.npy'
captions_path = os.path.join('./flickr30k', 'results_20130124.token')


def get_caption_data(captions_path, features_path):
     feats = np.load(features_path)
     captions = pd.read_table(captions_path, sep='\t', header=None, names=['image', 'caption'])
     captions = captions['caption'].values

     return feats, captions

# Andrej Karpathy NeuralTalk function
def preProBuildWordVocab(sentence_iterator, word_count_threshold=30):
    print 'preprocessing word counts and creating vocab based on word count threshold %d' % (word_count_threshold)
    word_counts = {}
    nsents = 0
    for sent in sentence_iterator:
      nsents += 1
      for w in sent.lower().split(' '):
        word_counts[w] = word_counts.get(w, 0) + 1
    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print 'filtered words from %d to %d' % (len(word_counts), len(vocab))

    idx2wrd = {}
    idx2wrd[0] = '.'  # END token
    wrd2idx = {}
    wrd2idx['#START#'] = 0 # START token
    ix = 1
    for w in vocab:
      wrd2idx[w] = ix
      idx2wrd[ix] = w
      ix += 1

    word_counts['.'] = nsents
    initial_bias = np.array([1.0*word_counts[idx2wrd[i]] for i in idx2wrd])
    initial_bias /= np.sum(initial_bias) # normalization
    initial_bias = np.log(initial_bias)
    initial_bias -= np.max(initial_bias)
    
    return wrd2idx, idx2wrd, initial_bias



features, captions = get_caption_data(captions_path, features_path)
wrd2idx, idx2wrd, initial_bias = preProBuildWordVocab(captions)

np.save('idx2wrd', idx2wrd)

index = np.arange(len(features))
np.random.shuffle(index)

features = features[index]
captions = captions[index]

sess = tf.InteractiveSession()
n_words = len(wrd2idx)
maxlen = np.max( map(lambda x: len(x.split(' ')), captions) )
caption_generator = cap_gen(
        dim_image=dim_image,
        dim_hidden=dim_hidden,
        dim_embed=dim_embed,
        batch_size=batch_size,
        n_lstm_steps=maxlen+2,
        n_words=n_words,
        initial_bias=initial_bias)

loss, image, sentence, mask = cap_gen.build_model()

saver = tf.train.Saver(max_to_keep=50)
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
tf.initialize_all_variables().run()

for epoch in range(n_epochs):
    for start, end in zip(range(0, len(features), batch_size), range(batch_size, len(features), batch_size)):

        current_feats = features[start:end]
        current_captions = captions[start:end]

        current_caption_ind = map(lambda cap: [wrd2idx[word] for word in cap.lower().split(' ')[:-1] if word in wrd2idx], current_captions)

        current_caption_matrix = sequence.pad_sequences(current_caption_ind, padding='post', maxlen=maxlen+1)
        current_caption_matrix = np.hstack( [np.full( (len(current_caption_matrix),1), 0), current_caption_matrix] ).astype(int)

        current_mask_matrix = np.zeros((current_caption_matrix.shape[0], current_caption_matrix.shape[1]))
        nonzeros = np.array( map(lambda x: (x != 0).sum()+2, current_caption_matrix ))

        for ind, row in enumerate(current_mask_matrix):
            row[:nonzeros[ind]] = 1

        _, loss_value = sess.run([train_op, loss], feed_dict={
            image: current_feats,
            sentence : current_caption_matrix,
            mask : current_mask_matrix
            })

        print "Current Cost: ", loss_value

    print "Epoch ", epoch, " is done. Saving the model ... "
    saver.save(sess, 'model', global_step=epoch)
    learning_rate *= 0.95