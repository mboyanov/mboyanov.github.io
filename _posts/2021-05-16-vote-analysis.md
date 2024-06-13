# K-means Clustering for Vote Analysis

On the 4th April 2021 we held elections in Bulgaria to choose the new parliament. This article will attempt to apply K-means clustering on the results and reason about the implications of the calculated clustering.

I view the results for a given region as a mixture of political opinions. It is naive to think that all sections voted uniformly according to the country average to arrive at the final result. On the other extreme, I find it unbelievable that all sections voted uniquely. By applying a clustering method, such as k-means we will be able to **identify common patterns** and reason about their implications.

The clustering results will give us the general trends, but it is also interesting to **examine the outliers** — what is the case with sections that are far away even from their closest cluster? These could be examples of voter fraud or of emerging new players in the political spectrum.

## The Data

The data is freely available from the Central Electoral Commission website: https://results.cik.bg/pi2021/csv.html
After some slight massaging, we form up a dataset where each row is a voting section and each column represents the total number of votes for each of the 30 parties that participated in the elections.
In order to ignore the relative number of voters that voted in each section, we normalize by the L1 norm. This way we will be modeling the percentage of voters in each section that voted for a given party. In a way, each section would be representing a probability distribution. Using clustering, we are interested in finding which groups of sections had similar probability distributions and hence voted in a similar manner.

Side note: in the data provided by the Central Electoral Commission, for each section and for each party we get three numbers: the total number of votes for the given party, the number of votes given with paper and the number of votes given by machine. In 50 of these cases these numbers do not add up. Please see [here](https://drive.google.com/file/d/1iLTuEFwvYq18gYr0PLvHKCn48rSO6c8U/view?usp=sharing) which are the problematic entries.


## K-Means Clustering in a Nutshell
1. Initialize the cluster centers
2. Until convergence:
   1. assign each point to the nearest cluster center 
   2. recompute the cluster centers as the means of all points assigned to the in the current step


The resulting cluster centers can then be interpreted as representatives for their respective clusters and we can reason about their significance.

## Application
30 parties participated in the elections. The country-wide results were as shown below. The parties that ended up being elected are with the following numbers(also highlighted):

| Number | Party |
|--------|-------|
| 4      | BSP   |
| 9      | DPS   | 
| 11     | DB    | 
| 18     | ISMV  | 
| 28     | GERB  | 
| 29     | ITN   | 

<img src="/images/votes1.webp" style="margin-left:-25%;margin-right:-25%;max-width:150%;"/>
<div class="figcaption">Actual results from the vote</div>

Specifying 7 clusters, we get the following cluster centers. Note how most cluster centers are quite far off from the country average. In the next sections we will comment briefly on each of the cluster centers and how we can interpret them.

For detailed participation in each cluster, please check the accompanying [Binder](https://mybinder.org/v2/gh/mboyanov/vote-analysis/HEAD?urlpath=voila%2Frender%2FClusterVotes.ipynb) demo.

<img src="/images/votes2.webp" style="margin-left:-25%;margin-right:-25%;max-width:150%;"/>
<div style="figcaption">Cluster centers for the 7 clusters</div>

Note that each cluster has a different number of sections that belong to it. In fact, the distribution is as follows:

1. Cluster 5 — 4816
2. Cluster 0 — 2647
3. Cluster 4 — 1416
4. Cluster 1 — 1104
5. Cluster 6 — 1049
6. Cluster 3—984
7. Cluster 2—925


Let’s analyze them in order of decreasing size.

## Cluster 5
Having the largest number of sections, this cluster is also closest to the country average. The most notable exceptions are that in these sections, the results for GERB and ITN are tied and that the support for DPS is non-present.

## Cluster 0
In this cluster, we have a clear win for GERB with 36% while the parties of the protest gain a total of 15+5+4 ~ 24%.

## Cluster 4
In cluster 4, Democratic Bulgaria takes the lead with a 28% score — 3 times higher than the national average. It’s intriguing to check which sections are the best representatives and which are furthest from this mindset.

## Cluster 1
Cluster 1 shows a great political divide. In these sections 79% of the vote went for DPS!

## Cluster 6
This cluster is another example where we see DPS win 41% of the vote and GERB following with 26%.

## Cluster 3
Cluster 3 represents the core GERB sections. Anyone claiming that GERB is going away for good, should really talk to these people.

## Cluster 2
Cluster 2 represents the core BSP sections with 41% — almost 3 times their national result.

## Cluster X — Where is Slavi?
Sometimes it helps to reason about what is missing. In this case we didn’t see a distinct cluster where ITN is the clear winner. Instead, we get cluster 5 where they are tied with GERB. One can argue that ITN is getting some level of support from all clusters and has the potential to increase it significantly in the next elections.

## Interpreting the outliers
Now that we have identified the major trends, we can focus on the outliers. What sections are the furthest from their closest cluster? These will be sections that deviate from the norm.

<img src="/images/votes3.webp" style="margin-left:-25%;margin-right:-25%;max-width:150%;"/>
<div style="figcaption">Vote distribution for the outliers — the sections that are furthest apart even from their closest cluster center</div>


As we can see, in most cases the outliers voted for parties that did not end up in the parliament — VMRO(1), Republicans for Bulgaria(21) or VOLQ/NFSB(24) — or had an unusually high score for BSP(4) or ITN(18). If we had specified a larger number of clusters, we would had probably gotten a corresponding cluster for these smaller players.

## Summary
We applied K-means to discover common patterns in the voting distributions. We identified 7 clusters with quite different characteristics representing the backbones of different parties or the main conflicts.

Additionally, we explored the outliers that were far way from even their closest centers and we found out that they represent the unrepresented — parties that did not enter the parliament.

## Links
[Voila Demo](https://mybinder.org/v2/gh/mboyanov/vote-analysis/HEAD?urlpath=voila%2Frender%2FClusterVotes.ipynb) (seems to be slow to load)    
[Github](https://github.com/mboyanov/vote-analysis)    
[Binder](https://mybinder.org/v2/gh/mboyanov/vote-analysis/HEAD)

