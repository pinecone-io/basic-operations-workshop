# Pinecone Basic Operations Workshop
This workshop contains a collection of labs that will get you hands-on with Pinecone for basic operations. If you would like more use case specific material
please refer to [Pinecone Examples](https://docs.pinecone.io/page/examples). If you are interested in deep dives, refer to [Pinecone Learning Center](https://www.pinecone.io/learn/)

## [Lab #1](./lab1.ipynb)  
This is a simple lab that has the basics for setting environment variables, installing dependencies and working with the pinecone client. This lab uses dummy data. 

## [Lab #2](./lab2.ipynb)  
This lab introduces the following operations concepts:

* Index creation using the x2 pod size (a way to vertically scale your index)
* Use meta-data statistics to learn more about the index
* Index backup/restore using Collections
* Meta-data filter exclusions to guard against high cardinality

This lab also uses dummy data. 

## [Lab #3](./lab3.ipynb)  

This lab introduces the following operations concepts:

* Load public datasets from Hugging Face
* Generate embeddings using an open source model(CLIP)
* Use training data labels as meta-data for your training data embeddings
* Query pinecone with a test image that is not included in the public data set
* Validate pinecone accuracy with test images that are included in the public data set 
* Run a load test using public test data to validate accuracy, P50-P100 latency and QPS