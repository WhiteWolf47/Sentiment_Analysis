# Importing libraries
import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import textstat
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Sentiment Analysis
def sentiment_analysis(text, pos, neg, total_words):

    # Count the positive and negative words
    positive_score = 0
    negative_score = 0
    for word in text:
        if word in pos:
            positive_score += 1
        if word in neg:
            negative_score += 1
    #negative_score = negative_score*-1
    polarity_score = (positive_score - negative_score)/((positive_score + negative_score)+0.000001)
    subjectivity_score = (positive_score + negative_score)/(total_words + 0.000001)
    return positive_score, negative_score, polarity_score, subjectivity_score

# Analysis of Redability
def readability(text):
    # Count the number of words
    tokens = regexp.tokenize(text)
    total_words = len(tokens)
    # Count the number of sentences
    sentences = nltk.sent_tokenize(text)
    total_sentences = len(sentences)
    # calculate average sentence length
    average_sentence_length = total_words/total_sentences
    # count number of complex words
    complex_words = 0
    for word in tokens:
        if textstat.flesch_reading_ease(word) < 100:
            complex_words += 1
    # calculate percentage of complex words
    percentage_complex_words = complex_words/total_words

    #calculate fog index
    fog_index = 0.4*(average_sentence_length + percentage_complex_words)
    
    return average_sentence_length, percentage_complex_words, fog_index, average_sentence_length, complex_words

 # Word Count   
def word_count(text):
    # Count the number of words
    tokens = regexp.tokenize(text)
    cleaned_words = [word for word in tokens if word.lower() not in stopwords]
    total_cleaned_words = len(cleaned_words)
    return total_cleaned_words

# Syllable count per word
def syllable_count(text):
    # Count the number of words
    tokens = regexp.tokenize(text)
    total_syllables = 0
    for word in tokens:
        total_syllables += textstat.syllable_count(word)

    syllble_count_per_word = total_syllables/len(tokens)
    return syllble_count_per_word 

# Personal Pronouns
def personal_pronouns(text):
    # Count the number of words
    tokens = regexp.tokenize(text)
    personal_pronouns = 0
    for word in tokens:
        if word in ['I', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours', 'you', 'your', 'yours', 'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them', 'their', 'theirs']:
            personal_pronouns += 1
    return personal_pronouns

# Average Word Length    
def avg_word_length(text):
    # Count the number of words
    tokens = regexp.tokenize(text)
    total_characters = 0
    for word in tokens:
        total_characters += len(word)
    average_word_length = total_characters/len(tokens)
    return average_word_length

# Main Function
if __name__ == '__main__':

    # defining list of my own custome spotwords
    my_stopwords = ['introduction']

    # Positive Word List
    pos_words = []
    with open("MasterDictionary/positive-words.txt", 'r') as f:
        for line in f:
            pos_words.append(line.strip())
    #print(pos_words)

    # Negative Word List
    neg_words = []
    with open("MasterDictionary/negative-words.txt", 'r') as f:
        for line in f:
            neg_words.append(line.strip())

    # Stopwords
    stopwords = stopwords.words("english")

    # Reading the data
    df = pd.read_csv('Input.xlsx - Sheet1.csv')

    # Adding the Output/Text column, Note - here text is referred to as Output
    df["Output"] = ""

    # Reading the articles text data we extracted from the input links using the scraper.py script
    for i in range(len(df)):
        with open(f"articles/{df['URL_ID'][i]}.txt", 'r') as f:
            df['Output'][i] = f.read()
        f.close()

    # Removing the stopwords and storing the cleaned text in a new column called "nso"
    df['nso'] = df['Output'].apply(lambda x: ' '.join([word.lower() for word in str(x).split() if word.lower() not in (stopwords) and word.lower() not in (my_stopwords)]))

    # Positive and Negative Word dictionaries
    df['pos'] = df['nso'].apply(lambda x: [word for word in str(x).split() if word.lower() in (pos_words)])
    df['neg'] = df['nso'].apply(lambda x: [word for word in str(x).split() if word.lower() in (neg_words)])

    # Tokenizer
    regexp = RegexpTokenizer(r'\w+')

    # Tokenizing the text
    df['text_token']=df['nso'].apply(regexp.tokenize)

    # Lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Lemmatizing the text
    df['text_lemmatized'] = df['text_token'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

    # Total Words
    df['total_words'] = df['text_lemmatized'].apply(lambda x: len(x))

    # Sentiment Analysis
    df['POSITIVE SCORE'], df['NEGATIVE SCORE'], df['POLARITY SCORE'], df['SUBJECTIVITY SCORE'] = zip(*df.apply(lambda x: sentiment_analysis(x['text_lemmatized'], pos_words, neg_words, x['total_words']), axis=1))

    # Readability, avgerage number of words per sentence, complex words count
    df['AVG SENTENCE LENGTH'], df['PERCENTAGE OF COMPLEX WORDS'], df['FOG INDEX'], df['AVG NUMBER OF WORDS PER SENTENCE'], df['COMPLEX WORD COUNT'] = zip(*df.apply(lambda x: readability(str(x['Output'])), axis=1))

    # Word Count
    df['WORD COUNT'] = df['Output'].apply(lambda x: word_count(str(x)))

    # Syllable per wordCount
    df['SYLLABLE PER WORD'] = df['Output'].apply(lambda x: syllable_count(str(x)))

    # Personal Pronouns
    df['PERSONAL PRONOUNS'] = df['Output'].apply(lambda x: personal_pronouns(str(x)))

    # Average Word Length
    df['AVG WORD LENGTH'] = df['Output'].apply(lambda x: avg_word_length(str(x)))

    # Writing the output to a csv file
    output_df = df[["URL_ID", "URL", "POSITIVE SCORE", "NEGATIVE SCORE", "POLARITY SCORE", "SUBJECTIVITY SCORE", "AVG SENTENCE LENGTH", "PERCENTAGE OF COMPLEX WORDS", "FOG INDEX", "AVG NUMBER OF WORDS PER SENTENCE", "COMPLEX WORD COUNT", "WORD COUNT", "SYLLABLE PER WORD", "PERSONAL PRONOUNS", "AVG WORD LENGTH"]]
    output_df.to_csv('outputs.csv', index=False)

    # Printing the output
    print(output_df)