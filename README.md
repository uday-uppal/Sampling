# Assignment 3 Sampling

### Discussion

Sampling is the process of selecting subset of a population for the purpose of gathering information about the population. Calculating the information on a sample than the whole population helps lower the computation cost and time taken. 

We took the following sampling techniques:-

1)**Simple Random Sampling** : is a type of probability sampling in which the we randomly selects a subset from the population. The code snippet , taken from 102053008.py file is as follows:-

    n_sample_simpleRandomSampling=(pow(1.96,2))*0.5*(1-0.5)/(pow((0.05),2))
    data_simpleRandomSampling=(data2.sample(int(n_sample_simpleRandomSampling),random_state=10))

2)**Systematic Sampling** : is a probability sampling method where we select members of the population at a regular interval.The code snippet implementing Systematic Sampling, taken from 102053008.py file is as follows:-

    data_systematic=data2.iloc[[i for i in range(5,1000,3)],:]

3)**Stratified Sampling** : In Stratified Sampling the population can be partitioned into subpopulations, and the samples are randomly selected to equaly represent each startum. The code snippet implementing Stratified Sampling, taken from 102053008.py file is as follows:-

    n_sample_StratifiedSampling=(pow(1.96,2))*0.3*(1-0.3)/(pow((0.05/2),2))
    data_stratified=data2.groupby('Class', group_keys=False).apply(lambda x: x.sample(int(n_sample_StratifiedSampling/2),random_state=11))
    
4)**Cluster Sampling** : is a sampling technique used when mutually homogeneous yet internally heterogeneous groupings are evident in a statistical population. Here the population is divided into clusters which are then selected at random. Each cluster should ideally be a representative of the population. The code snippet implementing Cluster Sampling, taken from 102053008.py file is as follows:-

    n_sample_ClusterSampling=(1.96**2)*0.1*(1-0.1)/((0.05/3)**2)
    s=set(list(data2['Time']))
    s1=pd.Series(list(s))
    data_clustered=(data2[data2['Time'].isin([ i for i in s1.sample(int(n_sample_ClusterSampling/3),random_state=20)])])
    
5)**Quota Sampling** : is a method for selecting sample from population where each subclass has been allocated a particular quota. This method has high chances of sampling bias.

    data_only0=data2[data2['Class']==0].iloc[:500]
    data_only1=data2[data2['Class']==1].iloc[:300]
    data_quotasampling =pd.concat([data_only0 ,data_only1], axis=0)

### Models

We used the following Machine Learning models:-

1)Logistic Regression

2)Decision Tree Classifier

3)Random Forest Classifier

4)KNeighbors Classifier

5)Gaussian Naive Bayes   

### Results

Simple Random Sampling      : [0.9090909090909091, 0.961038961038961, 0.974025974025974, 0.7922077922077922, 0.9090909090909091]

Systematic Random Sampling  : [0.8656716417910447, 0.8955223880597015, 0.9402985074626866, 0.7164179104477612, 0.8208955223880597]

Stratified sampling         : [0.9031007751937985, 0.9573643410852714, 0.9961240310077519, 0.7945736434108527, 0.7906976744186046]

Cluster Sampling            : [0.9262295081967213, 0.9795081967213115, 0.9918032786885246, 0.8442622950819673, 0.8073770491803278]

Quota Sampling              : [0.9125, 0.9625, 0.9875, 0.8625, 0.88125]

Random Forest Classifier method with Stratified sampling gives the highest accuracy 0.9961240310077519
