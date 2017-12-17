import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Likelihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    words_xlengths = test_set.get_all_Xlengths()

    for guess_word in range(test_set.num_items):
        word_probs = {}
        best_guess_score = float('-inf')
        word_guess = None

        for word, model in models.items():
            try:
                X, length = words_xlengths[guess_word]
                guess_score = model.score(X, length)
                word_probs[word] = guess_score

                if guess_score > best_guess_score:
                    best_guess_score = guess_score
                    word_guess = word

            except:
                pass

        probabilities.append(word_probs)
        guesses.append(word_guess)

    # return probabilities, guesses
    return probabilities, guesses
