# coding=utf-8
import logging, sys, torch, os, time
from gensim.models import KeyedVectors

cur_time = time.strftime('%Y-%m-%d.%H:%M:%S')
log_path = './log/log.{}.txt'.format(cur_time)
logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
file_handler = logging.FileHandler(log_path)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(message)s"))
logger.addHandler(file_handler)

std_handler = logging.StreamHandler(sys.stderr)
std_handler.setLevel(logging.INFO)
std_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(message)s"))
logger.addHandler(std_handler)

cuda_num = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_num
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
random_seed = 666
logger.info('#label=20, warm-up,max epoch=100, cuda: #{}.'.format(cuda_num))

class Word2Vec():
    def __init__(self):
        self.data = None
    def get(self):
        if self.data is not None:
            return self.data
        else:
            word2vec_path = './data/GoogleNews-vectors-negative300.bin'
            self.data = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
            return self.data

word2vec = Word2Vec()

class Config():
    def __init__(self):
        self.TASK = None
        self.UNIT_TEST = False
        self.WARM_UP = 0
        self.TEST_INTERVAL = 5
        self.SAVE_INTERVAL = 100
        self.DETAIL_RESULT = False
        self.COLD = False

        self.ent_path = {
            'W0': ['./data/cache/init_ent_info.d7741a5cb5b9d4775c629869bf632f03.cache.pkl', './model.bk/e2v_model.95.ckpt'],
            # 'W1': ['./data/cache/init_ent_info.ebc0aeece785ebe7337afcc86c34d945.cache.pkl', './model.bk/e2v_model.W1.85.ckpt'],
            'W1': ['./data/cache/init_ent_info.ebc0aeece785ebe7337afcc86c34d945.cache.pkl', './model.bk/e2v_model.W1.100.norm.ckpt'],
            # 'W1': ['./data/cache/init_ent_info.d7741a5cb5b9d4775c629869bf632f03.cache.pkl', './model.bk/e2v_model.95.ckpt'],
            # 'IE': ['./data/cache/init_ent_info.115bc3202d483113196216cc3c2f0f33.cache.pkl', './model.bk/e2v_model.IE.95.ckpt'],
            # 'IE': ['./data/cache/init_ent_info.0c010fd4b292d9abc6db5d35826f7d33.cache.pkl', './model.bk/e2v_model.IE.95.ckpt'],
            'IE': ['./data/cache/init_ent_info.0c010fd4b292d9abc6db5d35826f7d33.cache.pkl', 'model.bk/e2v_model.IE.95.norm.ckpt'],
            'IER': ['./data/cache/init_ent_info.e5a520e3b5a11d847ff39d1ce4157c34.cache.pkl', './model.bk/e2v_model.IER.85.ckpt'],
        }
        
#############################################################################################
#                               Ent-to-Vec HyperParameters
        self.BATCH_SIZE = 500
        self.NUMBER_BATCHES_PER_EPOCH = 4000
        self.LEARNING_RATE = 0.3
        self.NUM_WORDS_PER_ENT = 20
        self.NUM_NEG_WORDS = 5
        self.HYP_CTXT_LEN = 10
        self.EMBEDDING_SIZE = 300
        self.MAX_EPOCH = 5000
#############################################################################################
#                               Entity Linking HyperParameters
        self.EL_CTXT_LEN = 100
        self.EL_R = 50
        self.EL_MAX_CAND_NUM = 30
        self.EL_MAX_KNOWN_ENT_NUM = 30
        self.EL_BATCH_SIZE = 32
        self.EL_LEARNING_RATE = 1e-4
        self.EL_MAX_EPOCH = 601
        self.EL_F_NETWORK_SIZE = 100
#############################################################################################
        self.LOCAL_E2V_CTXT_LEN = 50
        self.LOCAL_E2V_EPOCH = 15
        self.LOCAL_E2V_LR = 1e-2

    def ent_utils_path(self):
        task = self.TASK
        if task and task in self.ent_path:
            return self.ent_path[task][0]
        else:
            return None

    def ent_model_path(self):
        task = self.TASK
        if task and task in self.ent_path:
            return self.ent_path[task][1]
        else:
            return None

config = Config()

