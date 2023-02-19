# Assignment 3 Sampling

### Discussion

Sampling is the process of selecting subset of a population for the purpose of gathering information about the population. Calculating the information on a sample than the whole population helps lower the computation cost and time taken. 

We took the following sampling techniques:-

1)Simple Random Sampling : is a type of probability sampling in which the we randomly selects a subset from the population. The code snippet , taken from 102053008.py file is as follows:-

    n_sample_simpleRandomSampling=(1.96**2)*0.5*(1-0.5)/(0.05**2)
    data_simpleRandomSampling=(data2.sample(int(n_sample_simpleRandomSampling),random_state=10))

2)Systematic Sampling
3)Stratified Sampling
4)Cluster Sampling
5)Quota Sampling
