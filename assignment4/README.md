## Preparing data

In [BuddyMove Dataset Data Set ](https://archive.ics.uci.edu/ml/datasets/BuddyMove+Dataset#) was populated from 
destination reviews published by 249 reviewers of holidayiq.com till October 2014. Reviews falling in 6 categories
 among destinations across South India were considered and the count of reviews in each category for every reviewer
  (traveler) is captured.


## Attribute Information
* Attribute 1 : Unique user id 
* Attribute 2 : Number of reviews on stadiums, sports complex, etc. 
* Attribute 3 : Number of reviews on religious institutions 
* Attribute 4 : Number of reviews on beach, lake, river, etc. 
* Attribute 5 : Number of reviews on theatres, exhibitions, etc. 
* Attribute 6 : Number of reviews on malls, shopping places, etc. 
* Attribute 7 : Number of reviews on parks, picnic spots, etc.

# Analysis

## Silhouette (clustering)
Silhouette refers to a method of interpretation and validation of consistency within clusters of data.
 The technique provides a succinct graphical representation of how well each object lies within its cluster.

The silhouette value is a measure of how similar an object is to its own cluster (cohesion) compared to other clusters (separation).
 The silhouette ranges from −1 to +1, where a high value indicates that the
 object is well matched to its own cluster and poorly matched to neighboring clusters. If most objects have a high value,
  then the clustering configuration is appropriate. If many points have a low or negative value, then the clustering configuration may have too many or too few clusters.

The silhouette can be calculated with any distance metric, such as the Euclidean distance or the Manhattan distance.

## Kmeans

    silhouette_score : 0.314226159398104


## Hierarchical clustering 

    linkage : ward
    silhouette_score : 0.24418093101151334
    
    linkage : average
    silhouette_score : 0.4763807066815255
    
    linkage : single
    silhouette_score : 0.4763807066815255
    
    linkage : complete
    silhouette_score : 0.4763807066815255

