## Fake News Detection/ News Verifiability

This project is intended to combat fake news using stance detection. The model takes in as input a headline/claim and the corresponding article body. The output can be any of the 4, _Agree_ , _Discuss_ , _Disagree_ or _Unrelated_ , depending upon the relation between the claim and the body.

## How To Use 
* `$ git clone`
* `$ python server.py`
* `$ python request.py`
* `$ python sources_request.py`

## Model
* The current model is a basic multi-layer perceptron which exploits TF-IDF and Term frequency features. The input to the network is   feature vector of shape (10001,) which include headline features (5000,), the cosine-similarity between TF-IDF scores of headline and body (1,) and the body features (5000,). 
* The hidden layer consists of 100 units followed by a dropout of 0.6. The output layer is a softmax layer of 4 units corresponding to each possible class. ReLU activation function is used in the layers. The model can be further improved by using LSTMs and word-embeddings and is on the to-do list.

## Information Retrieval
* Given the headline/claim, we extract keywords from the headline using _Named-Entity-Recognition_ and _POS_ tags. This is done using        _spacy_. This method of keyword extraction is to be improved.
* These keywords are used to search/crawl the web for similar articles, i.e articles that have similar keywords. Event Registry is used     for this. It allows us to fetch articles based on keywords (15 at max).
* The article bodies are extracted from the fetched articles. Stance detection is performed on all these bodies with the given headline/claim. This gives us rough idea of how many different/other sources agree/disagree with the given claim.

## Screenshots
* Given claim and body, what is the relation between them.

![output2](https://user-images.githubusercontent.com/32245327/54678720-bc84af80-4b2b-11e9-9488-9fa07a07cfff.JPG)

* The first line lists the keywords being used for the search, next 5 lines list down the relevant urls of the articles fetched and the last line outputs the result of stance detection for each of the articles. 

![image](https://user-images.githubusercontent.com/32245327/54678645-92cb8880-4b2b-11e9-9948-3fbaec799a4a.png)

