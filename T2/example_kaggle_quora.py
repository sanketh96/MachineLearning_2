"""
This module is sample code that shows how to invoke the web service to get datasets
Author: Palacode Narayana Iyer Anantharaman
Date: 5 Oct 2018
"""
from nltk import sent_tokenize, word_tokenize
from api_client import ApiClient

if __name__ == "__main__":
    # in order to use the web service, first create the instance of ApiClient class
    client = ApiClient()

    # test it with echo service so that we are sure that the web service is running
    val = client.echo("hi there!!!!")
    print("The Service Returned: ", val)

    # you can use the method get_amazon_product_reviews to get data for sentiment analysis
    # this will be rate limited to 500 samples per call
    num_samples = 5
    val = client.get_kaggle_quora_data(num_samples)

    # you can print the variable returned by the service and examine the output fields
    print(val)

    # you can access individual fields such as "summary" and "rating" that can be used
    # to find the sentiment
    for item in val[:10]:
        print(item)
    
    # question1 may be one or more sentences of text. We need to break these in to words
    # further, we need to convert each word to a vector form
    # our web service provides a function that accepts a list of words and returns the
    # corresponding vectors. In the example below, we take the first item returned by the
    # previous call and convert that in to a sequence of vectors
    text = val[0]["question1"]
    print("The input text is: ", text)

    ### IMPORTANT NOTE: The code below shows how to get glove vectors for a list of words
    ### As the same word might repeat many times in the dataset you fetched above, you
    ### should AVOID invoking the API with the question sentences. Instead, you should
    ### gather all the data in text form as above and using that as a corpus, form a 
    ### vocabulary V. Then, create a python list of vocabulary (let's call it vocab) and 
    ### get the vectors for the vocab. If the glove API is rate limited, split your vocab
    ### in to multiple parts and invoke it repeatedly. Once you get all the required vectors
    ### you can form an embedding matrix and invoke it as Keras embedding layer

    # get sentence tokens from text that may have more than 1 sentence
    # we use NLTK's sent_tokenize for this
    sentences = sent_tokenize(text) # we get the list of sentences
    all_words = []
    for sentence in sentences:
        all_words.extend(word_tokenize(sentence))
    
    # all_words contains all the words in the text as a single list
    # let us get the vectors for these
    ### THE FOLLOWING IS FOR ILLUSTRATION ONLY - DON'T USE IT THIS WAY - see my comments above
    vals = client.w2v(all_words)
    for val in vals:
        print(val["word"], val["vec"])

    # now you can continue further by vectoring the class label and creating the required dataset
    # your code ......
