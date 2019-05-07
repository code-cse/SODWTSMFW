from collections import namedtuple
from data import Vocab
from model import SummarizationModel
import beam_search
from batcher import Example
from batcher import Batch
import tensorflow as tf
import data
import subprocess

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean('pointer_gen', True, 'If True, use pointer-generator model. If False, use baseline model.')
tf.app.flags.DEFINE_boolean('coverage', True, 'Use coverage mechanism.')
tf.app.flags.DEFINE_boolean('single_pass', True, 'For decode')

tf.app.flags.DEFINE_integer('hidden_dim', 256, 'dimension of RNN hidden states')
tf.app.flags.DEFINE_integer('emb_dim', 128, 'dimension of word embeddings')

tf.app.flags.DEFINE_integer('max_enc_steps', 2000, 'max timesteps of encoder (max source text tokens)')
tf.app.flags.DEFINE_integer('max_dec_steps', 100, 'max timesteps of decoder (max summary tokens)')
tf.app.flags.DEFINE_integer('beam_size', 4, 'beam size for beam search decoding.') 
tf.app.flags.DEFINE_integer('batch_size', 16, 'minibatch size')
tf.app.flags.DEFINE_integer('min_dec_steps', 35, 'Minimum sequence length of generated summary. Applies only for beam search decoding mode')



tf.set_random_seed(1111)
FLAGS.batch_size = FLAGS.beam_size 
# size of summary - max 1/25 input

HParams = namedtuple('HParams','emb_dim max_enc_steps pointer_gen max_dec_steps mode batch_size cov_loss_wt lr trunc_norm_init_std coverage max_grad_norm rand_unif_init_mag adagrad_init_acc hidden_dim')
hps = HParams(emb_dim=128, max_enc_steps=3000, pointer_gen=True, max_dec_steps=1, mode='decode', batch_size=4, cov_loss_wt=1.0, lr=0.15, trunc_norm_init_std=0.0001, coverage=True, max_grad_norm=2.0, rand_unif_init_mag=0.02, adagrad_init_acc=0.1, hidden_dim=256)
vocab = Vocab('vocab', 50000)

import time	


def get_config():
	"""Returns config for tf.session"""
	config = tf.ConfigProto(allow_soft_placement=True)
	config.gpu_options.allow_growth=True
	return config

tic = time.time()
# Building graph and loading kcheckpoint
hps = hps._replace(max_dec_steps=1)
model = SummarizationModel(hps, vocab)
model.build_graph()
saver = tf.train.Saver()
sess = tf.Session(config=get_config())
saver.restore(sess, 'train/model-238410')

print(time.time() - tic,"==============================")



def summary(input, sess, model, vocab, hps, prop):
	""" Returns decoded output for input
	input - tokenized, (lowercased) input
	sess : tensorflow session with params from checkpoint
	model - object of summarization model
	vocab - vocabulary file with id
	hps - hyperparameters 
	prop - proportion of the summary (prop times less)
	"""

	# reading file, tokenizing and creating batch for beam search
	FLAGS.min_dec_steps =  10 # summary at least 1/25 size of input
	FLAGS.max_dec_steps = 40
	command = ['java', '-cp', '../stanford-corenlp-full-2018-02-27/stanford-corenlp-3.9.1.jar', 'edu.stanford.nlp.process.PTBTokenizer', '-preserveLines']  #, '-lowerCase'
	ps = subprocess.Popen(('echo', input), stdout=subprocess.PIPE)
	article = subprocess.check_output(command, stdin=ps.stdout)
	ps.wait()

	example = Example(article, article, vocab, hps)
	b = [example for _ in xrange(hps.batch_size)]
	batch = Batch(b, hps, vocab)

	# beam search decoder
	best_hyp = beam_search.run_beam_search(sess, model, vocab, batch)
	output_ids = [int(t) for t in best_hyp.tokens[1:]]
	decoded_words = data.outputids2words(output_ids, vocab, (batch.art_oovs[0] if FLAGS.pointer_gen else None))

      # Remove the [STOP] token from decoded_words, if necessary
	try:
		fst_stop_idx = decoded_words.index(data.STOP_DECODING) # index of the (first) [STOP] symbol
		decoded_words = decoded_words[:fst_stop_idx]
	except ValueError:
		decoded_words = decoded_words
	
	decoded_output = ' '.join(decoded_words)

	
	#print 'Summary',decoded_output
	return decoded_output






def summary_text(path):
	f = open('results/summary.txt','w')

	with open(path) as f_in:
		lines = [line.rstrip() for line in f_in]
		lines = [line for line in lines if line]
	articles = lines

	
	for i in range(1):
		
		print ""
		print "GENERATED HEADLINE:"
		gen_head = summary(articles[i], sess, model, vocab, hps, 20)
		# print("============gen head=================")
		# print(gen_head)
		mod_gh = gen_head.split('.')
		# print("===========mod_gh======================")
		# print(mod_gh)

		if len(mod_gh) > 1:
			mod_gh = mod_gh[0]
			final_gh = mod_gh+'.'
		else:
			final_gh = gen_head
		print final_gh
		print ""
		f.write("----------------"+str(i+1)+"----------------\n")
		f.write("GENERATED HEADLINE:\n")
		f.write(final_gh)
		f.write("\n\n")
	f.close()


path = "test_data/news2.txt"
# path = "/home/ashok/Desktop/NLP/text_summarisation_using_pointer_generator/src/ptr_gnrtr/test_data/news9.txt"

t = time.time()

summary_text(path)

print(time.time() - t,"--------------------------------new time-----------------")

