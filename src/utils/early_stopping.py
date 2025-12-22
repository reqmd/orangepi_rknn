import torch
import torch.nn as nn

class EarlyStopping():
    def __init__(self, min_delta = 0.001, patience = 5):
        self.min_delta = min_delta
        self.patience = patience
        self.counter = 0
        self.loss = None
    
    def step(self, loss_now, need_debug = False):
        if self.loss == None:
            self.loss = loss_now
        else:
            if need_debug == True:
                print(f'Изменение лосса: {self.loss - loss_now}')

            if self.loss - loss_now >= self.min_delta:
                self.counter = 0

            else:
                self.counter+=1
                if need_debug == True:
                    print('Увеличили счетчик на 1')
                    print(f'На данный момент счетчик: {self.counter}')
            self.loss = loss_now
        if self.counter >= self.patience:
            print('Ранняя остановка!')
            return 1
        else:
            return 0


