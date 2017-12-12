import numpy as np
from config import Config
from model import model_fn
import preprocessing as prep
from keras.callbacks import CSVLogger
from keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from keras.models import load_model


def run():

    config = Config()  # Load configs

    save_path = 'keras_models/keras_model'  # Model save path

    x_train_path = 'data/xtrain.txt'
    x_test_path = 'data/xtest.txt'
    y_train_path = 'data/ytrain.txt'

    x_idx = prep.Indexer()

    X = prep.read_file(x_train_path, raw=True)
    y = prep.read_file(y_train_path, label=True)

    t = CountVectorizer(analyzer='char', ngram_range=(config.ngram, config.ngram))
    t.fit(X)
    X = prep.transform(X, t, x_idx)
    X = np.array(pad_sequences(X, config.maxlen))

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=config.test_size, shuffle=config.shuffle)

    #############################################
    # Train model
    print("BEGINNING TRAINING")
    tsv_logger = CSVLogger('training-data.tsv', append=True, separator='\t')

    m = model_fn(config=config, input_length=x_idx.max_number()+1)

    # m = load_model(save_path)

    m.fit(x_train, y_train, epochs=config.n_epochs, batch_size=config.batch_size,
          verbose=1, shuffle=True, callbacks=[tsv_logger],
          validation_data=(x_test, y_test))

    m.save(save_path)

    print("MODEL REPORT")
    score, acc = m.evaluate(x_test, y_test)

    print("\nSCORE: ", score)
    print("ACCURACY: ", acc)

    pred = [np.argmax(label) for label in m.predict(x_test)]

    report = classification_report(y_test, pred)

    print(report)


    ###############################################
    # Predict and write labels for xtest.txt

    print("PREDICTION")

    X = prep.read_file(x_test_path, raw=True)
    X = prep.transform(X, t, x_idx, add_if_new=False)
    X = np.array(pad_sequences(X, config.maxlen))

    pred = [np.argmax(label) for label in m.predict(X)]

    with open("".join(["keras_prediction/ytest.txt"]), "w+", encoding="utf-8") as rec:
        for label in pred:
            rec.write("%s\n" % label)

        rec.close()


if __name__ == '__main__':
    run()
