import sys
import numpy as np
import tensorflow as tf
from model import read_image
from model import cap_gen

vgg_path = './VGG_ILSVRC_19_layers.tfmodel'
test_image_path = sys.argv[1]
model_path = "model-#" # Path to model
maxlen = 30

# Implementing Bruteforce search
with open(vgg_path) as file:
    content = file.read()
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(content)

imgs = tf.placeholder("float32", [1, 224, 224, 3])
tf.import_graph_def(graph_def, input_map={"images":imgs})

idx2wrd = np.load('./idx2wrd.npy').tolist()
n_words = len(idx2wrd)

image_val = read_image(test_image_path)
sess = tf.InteractiveSession()

graph = tf.get_default_graph()
fc7 = sess.run(graph.get_tensor_by_name("import/fc7_relu:0"), feed_dict={imgs:image_val})

cp_gen = cap_gen(dim_image=4096,
     	         dim_hidden=256,
		         dim_embed=256,
		         batch_size=128,
		         n_lstm_steps=maxlen,
		         n_words=n_words)

fc7_tf, gen_wrds = cp_gen.build_generator(maxlen=maxlen)

saver = tf.train.Saver()
saver.restore(sess, model_path)

gen_wrd_index= sess.run(gen_wrds, feed_dict={fc7_tf:fc7})
gen_wrd_index = np.hstack(gen_wrd_index)

gen_wrds = [idx2wrd[x] for x in gen_wrd_index]
stop = np.argmax(np.array(gen_wrds) == '.')+1

gen_wrds = gen_wrds[:stop]
gen_cap = ' '.join(gen_wrds)
print gen_cap