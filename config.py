class Config:
    n_epochs = 50

    batch_size = 128
    maxlen = 300
    char_embed_size = 128
    label_size = 12

    ngram = 7
    test_size = 0.1
    hidden_sizes = [128, 32, 64]
    shuffle = False
    
    activation = 'softmax'
    input_dropout = 0.3
    dropout = 0.2
