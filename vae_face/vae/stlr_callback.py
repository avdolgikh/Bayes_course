# source: https://qiita.com/ytkj/items/bd6207e562c1b1270d33
# https://translate.google.com/translate?hl=ru&sl=ja&u=https://qiita.com/ytkj/items/bd6207e562c1b1270d33&prev=search
# https://towardsdatascience.com/openai-gpt-language-modeling-on-gutenberg-with-tensorflow-keras-876f9f324b6c
# https://www.topbots.com/ai-nlp-research-pretrained-language-models/
# !!! https://www.pyimagesearch.com/2019/07/29/cyclical-learning-rates-with-keras-and-deep-learning/
# BERT fine-tuning https://arxiv.org/pdf/1905.05583.pdf

import keras
from keras import backend as K

class KerasSlantedTriangularLearningRateCallback(keras.callbacks.Callback):
    """
    The slanted triangular learning rate schedule used for ULMFiT https://arxiv.org/pdf/1801.06146.pdf
    """

    def __init__(self,
                 lr_max: float = 0.001,
                 cut_frac: float = 0.1,
                 ratio: float = 32):
        self.lr_max = lr_max
        self.cut_frac = cut_frac
        self.ratio = ratio
        self.lr_history = []

    def on_train_begin(self, logs = None):
        epochs = self.params['epochs']
        steps = int(self.params['samples'] // self.params['batch_size']) # self.params['steps']
        self.cut = epochs * steps * self.cut_frac
        self.iteration = 0

    def on_batch_begin(self, batch: int, logs = None):
        t = self.iteration
        cut = self.cut
        if t < cut:
            p = t / cut
        else:
            p = 1 - (t - cut) / (cut * (1 / self.cut_frac - 1))
        lr = self.lr_max * (1 + p * (self.ratio - 1)) / self.ratio
        
        K.set_value(self.model.optimizer.lr, lr)

        #self.lr_history.append( lr )
        self.lr_history.append( float(str(K.eval(self.model.optimizer.lr))) )

        self.iteration += 1


