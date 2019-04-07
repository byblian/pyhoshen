---
title: Forecasting the Israeli Elections using pymc3
---

Welcome!

Election polls are possibly the clearest most public example of statistics that we encounter in lives. We depend on them, as voters (or candidates), to understand where the election stands. They can contribute to tactical voting on the voters' part, and can influence decisions on the part of candidates and politicians. But what do they mean? How does the margin-of-error figure into the numbers? What are we to understand if two polls have conflicting results?

<!--more-->

Here in Israel, the issue is even more pronounced: It is 2019, and there are currently over a dozen parties that have some viable chance of receiving some seats in parliament (called the Knesset). But a whole bunch of them are trailing near the "threshold." The threshold - at 3.25% - means that if a party does not pass the threshold, its votes are divided up amongst the other parties using a complex algorithm called Bader-Ofer. Multiple polls taken at the same time might show different parties passing the threshold, and this naturally affects the final results.

This is the purpose of pyHoshen. pyHoshen is a framework for Bayesian MCMC election-polling written in pymc3. pymc3 is designed as an "Inference button" - just write the statistical model in high-level python and push the "Inference buttom" and it translates the code into a Bayesian MCMC model. Computing the model might take some time, but pymc3 uses C++ and the GPU to provide extra computational speed. Similarly, pyHoshen attempts to be the "Inference button" for election-polling - just feed in the polls and set up the election parameters, and it translates the polls into the statistical model for pymc3. This is a work-in-progress and contributions and comments are welcome!

It is important to understand - the model is a complex "poll of polls." Whereas other poll of polls might provide a simple average, pyHoshen can account for correlations between parties and potential house effects of pollsters.

* Correlations imply that the model might discern that two parties are negatively correlated. For example, they might be competing on the same electorate - if one party goes up, the corresponding party goes down and vice versa. This means that the results of the model will consider the correlations between the parties. If in a particular scenario that the model considers, the first party receives a higher support, the second will receive a lower support due to the correlations.

* Normal polling error - Polls have an inherent error associated with them. The model accounts for this error by using a multivariate Student-T test for the polls.

* House effects are the systematic tendencies of pollsters to give a better or worse percentage to a particular party beyond the normal polling error. We do not know if they are right or wrong in this, but we can determine that they systematically rank a party better or worse than the average. We use this to cancel out the "house effects" and determine where the average support for each party lies.

However, the model does not contain more information than the polls themselves. It is still just an elaborate poll of polls. If all the polls are wrong, or if the polls are on average wrong or off by some percentage points, the model will be wrong too.

## Setup

So how do we run it? We start with the following initialization block which clones the pyHoshen repository and sets up the environment. I also authenticate to google for google docs, in order to access the polls database which is hosted at [google sheets](https://bit.ly/polls2019). Finally, we check the uptime. Google Colab only allows 12 hours of use at a time, and running the model can take over an hour or more. So if the uptime is close to the limit, it might be a good idea to reset the runtimes.


```
!rm -rf pyhoshen
!git clone https://github.com/byblian/pyhoshen.git
!echo -n Current commit: & GIT_DIR=`pwd`/pyhoshen/.git git rev-parse HEAD

from google.colab import auth
auth.authenticate_user()

# setup hebrew-compatible fonts
!pip install python-bidi
import matplotlib
matplotlib.rc('font', family='DejaVu Sans')

!uptime
```

    Cloning into 'pyhoshen'...
    remote: Enumerating objects: 132, done.[K
    remote: Counting objects: 100% (132/132), done.[K
    remote: Compressing objects: 100% (87/87), done.[K
    remote: Total 132 (delta 76), reused 80 (delta 40), pack-reused 0[K
    Receiving objects: 100% (132/132), 46.61 KiB | 1.14 MiB/s, done.
    Resolving deltas: 100% (76/76), done.
    Current commit:f6f040d58e4ee6816cd924724758fdb83d6514ab
    Collecting python-bidi
      Downloading https://files.pythonhosted.org/packages/e0/a7/ea6334b798546ed8584cb8cdc5d153c289294287b4ab46e9a4242480eae3/python_bidi-0.4.0-py2.py3-none-any.whl
    Requirement already satisfied: six in /usr/local/lib/python3.6/dist-packages (from python-bidi) (1.11.0)
    Installing collected packages: python-bidi
    Successfully installed python-bidi-0.4.0
     18:25:18 up 1 min,  0 users,  load average: 0.47, 0.20, 0.08
    

Having initialized let's set up the model. We choose to forecast for today, which was March 17, 2019 at the time I ran it, and about 3 weeks ahead of the election. We use the "add-mean" model to account for house effects. This model specifies that each pollster has a certain percentage point for each party that is added to the actual party's mean. There are various house effects models, but this seems to be the most stable and allows the model to converge. Bayesian MCMC solves a particular statistical model by running random scenarios in a chain. Over time, we expect the model to converge and we can check this by running multiple chains and seeing if they arrived at the same results. We use a maximum of the 35 recent days of polls. The eta parameter is related to the correlation matrix that is central to the model, and corresponds a degree of correlations we expect in the final result.


```
import theano
theano.config.gcc.cxxflags = '-O3 -ffast-math -ftree-loop-distribution -funroll-loops -ftracer'
theano.config.allow_gc = False
theano.config.compute_test_value = 'off'

import datetime
import pymc3 as pm
from pyhoshen import israel
forecast_day = datetime.date.today()
with pm.Model() as model:
  election = israel.IsraeliElectionForecastModel(
      'https://drive.google.com/uc?id=1WYGgC3LeTkwKz0Oc2IYX5OdnyiroSA9P',
      house_effects_model='add-mean', base_elections=[], max_days=35,
      forecast_day=forecast_day, eta=25)
```

## Running the model

All good, so let's run it! The particular technical options below are useful to deal with some diagnostic warnings. In particular, "cores=2" forces google to run the two chains concurrently. Google Colab only has one CPU so pymc3 would normally run two chains sequentially by default. However, in my experience, "cores=2" runs faster (and possibly less buggy). In total, we will run 9,000 random scenarios or "draws" - 4500 on each of the two chains running concurrently. The first 2000 draws of each chain are used for tuning and discarded. They allow each chain to reach convergence to the final result.  The final 2500 samples from each chain are going to constitute our final results, for a total of 5000 samples. Each of this sample is a unique possible way the election campaign could have unfolded to provide us the polls we see.


```
!uptime
with model:
  print (theano.config.gcc.cxxflags)
  print (election.forecast_election)
  print (election.forecast_model.house_effects_model)
  print ("forecast_day", forecast_day)
  nuts_kwargs=dict(target_accept=.8, max_treedepth=25, integrator='three-stage')
  samples = pm.sample(2500, tune=2000, cores=2, nutws_kwargs=nuts_kwargs)
```

     18:27:04 up 3 min,  0 users,  load average: 1.09, 0.49, 0.20
    

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    

    -O3 -ffast-math -ftree-loop-distribution -funroll-loops -ftracer
    election21_2019
    add-mean
    forecast_day 2019-03-17
    

    WARNING (theano.tensor.blas): We did not find a dynamic library in the library_dir of the library we use for blas. If you use ATLAS, make sure to compile it with dynamics library.
    WARNING (theano.tensor.blas): We did not find a dynamic library in the library_dir of the library we use for blas. If you use ATLAS, make sure to compile it with dynamics library.
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [election21_2019_polls_pollster_house_effects, election21_2019_polls_innovations, election21_2019_votes, election21_2019_cholesky_pmatrix]
    Sampling 2 chains: 100%|██████████| 9000/9000 [1:39:12<00:00,  2.04draws/s]
    

The model took us 1:39 hours to complete! pymc3 will usually perform some diagnostics and warn us of them, if the model did not converge properly, for example. These warnings can imply errors or shortcomings in the model itself. But there are no warnings, meaning the model converged properly.

## Analyzing the results

The results we sampled only contain percentages of the various parties. As mentioned above, the Israeli parliament (Knesset) assigns seats based on an algorithm called Bader-Ofer, and election-polls are published using these Knesset seats. To convert the percentages we received to seats, we now compute the Bader-Ofer seat allocation for the entire trace. The trace is the set of all samples we got. Each of these samples contain a separate state of elections on each day of the election up to the forecast day. So we now run the Bader-Ofer algorithm separately on each of the 35 days of each of the 5000 samples. We do it now because this is a heavy computation that is not necessary during the main run and would just slow down the computations. The model is a statistical model and uses a multivariate Student-T distribution to model the polls, and this requires percentages, not parliament seats. 


```
import theano  
theano.config.compute_test_value = 'off'
support_samples = samples['support']
bo=election.compute_trace_bader_ofer(support_samples)
```

## Visualizing the results

### Mandates Graph

So now we have the results! What can we see? The most standard display of poll results in Israel is the mandates bar graph. So we can just plot these.


```
election.plot_mandates(bo, hebrew=False)
election.plot_mandates(bo, hebrew=True)
```


![png](/images/2019-03-17-Forecasting-the-Israeli-Elections-using-pymc3/2019-03-17-Forecasting-the-Israeli-Elections-using-pymc3_9_0.png)



![png](/images/2019-03-17-Forecasting-the-Israeli-Elections-using-pymc3/2019-03-17-Forecasting-the-Israeli-Elections-using-pymc3_9_1.png)


However, in addition to the normal bar graph that we are used to, we also see the distribution of each party's support. We can see, for example, that "New Yamin" has a 76% of passing the threshold. Zehut - a type of libertarian 420-friendly party that is surprisingly making headway in the polls has a 67% chance of passing the threshold. Similarly, Raam-Balad, a union of two Arab parties that was almost blocked from running due to associations with terrorism also has a 67% chance of passing the threshold. Now, these are displayed as individual bar-graphs but we have to remember that the model always considers correlations. Raam-Balad split from the "Joint Arab List" on the last day of party registration, leaving the other two Arab parties Hadash-Taal by themselves. Both of these party unions are competing primarily in the same electorate - the Arab population. So if in one modeled scenario, Raam-Balad receives a low amount of support, it is likely that Hadash-Taal would receive a higher amount of support in that same scenario.

Showing bar graphs for each party is nice but we also want to provide one final result. After the actual election, we might compare this final results to the true election election by determining the "error" it has -- how far it is from the real true election results. So to determine the sample that is going to be our displayed final result, we choose the sample whose average "error" to all the other samples is smallest. In this way, we are displaying a true sample from the 5000 samples we randomly drew, and in this samples all the correlations and house effects come together in a consistent way. The technical method for determining the "error" is to use the square root of the sum of squares of the differences of the party mandate results.

### Party Support over Time

But our model did not just compute the final results, it also computed the results over time. We can plot these as well.


```
election.plot_party_support_evolution_graphs(support_samples, bo, hebrew=False)
election.plot_party_support_evolution_graphs(support_samples, bo, hebrew=True)
```


![png](/images/2019-03-17-Forecasting-the-Israeli-Elections-using-pymc3/2019-03-17-Forecasting-the-Israeli-Elections-using-pymc3_11_0.png)



![png](/images/2019-03-17-Forecasting-the-Israeli-Elections-using-pymc3/2019-03-17-Forecasting-the-Israeli-Elections-using-pymc3_11_1.png)


In the above graphs, we might be able to see some trends such as Likud going up while New Yamin and Kulanu are going down. These are all parties competing on the right-wing electorate. We can also see that Blue & White, the main opposition party to the Likud did drop in support but has remained relatively stationary over the last few days, while Zehut -- the libertarian 420-friendly surprise party - has been constantly rising in support.

### Technical sample details

As an initial first step to understand the diagnostics of sample, I run the following. The summary is a table of statistics for each random variable in the model. For example, each party's percentage on each day is a separate random variable. Primarily, I look at the number of effective samples (n_eff) and Rhat. Rhat is the Gelman-Rubin statistic used to compare the results of the chains and should converge to 1 if the model converge. The n_eff indicates the the random variable with the lowest amount of effective samples has 1393 different effective samples, while the average effective samples in a variable is 6046. Similarly, we can see that the Gelman-Rubin statistic (Rhat) is 0.9998 at the lowest, 0.99979 on average, and 1.002911 at the maximum, all of which indicate good convergence.  


```
import pymc3 as pm
summary = pm.summary(samples)
print ("min", summary.min())
print ("max", summary.max())
print ("mean", summary.mean())
```

    min mean          -0.039167
    sd             0.000000
    mc_error       0.000000
    hpd_2.5       -0.064648
    hpd_97.5      -0.025353
    n_eff       1393.513021
    Rhat           0.999800
    dtype: float64
    max mean           1.000000
    sd             0.020796
    mc_error       0.000328
    hpd_2.5        1.000000
    hpd_97.5       1.000000
    n_eff       9018.285041
    Rhat           1.002911
    dtype: float64
    mean mean           0.070819
    sd             0.005880
    mc_error       0.000075
    hpd_2.5        0.059256
    hpd_97.5       0.082349
    n_eff       6046.975224
    Rhat           0.999979
    dtype: float64
    

### House Effects

The model also computed house effects so we can display these. We can see, for example, that "New Yamin" has a 0.73% extra house effect in "Direct Polls," who is also that party's private pollster. This translates to about an extra mandate (1/120 = 0.83%). Similarly, a pollster called Miskar that published results only twice (and of these only once with full information), gave Zehut 7 mandates when everyone else gave it none. We see their house effect for Zehut is 2.99% - about 3.6 mandates. So it seems the model sees Zehut as having had about 3.4 worth of mandates support at that time -- but unfortunately, that is below the threshold which is at 3.9 mandates (3.25% of 120 is 3.9), which would explain why everyone else gave it 0 mandates. Zehut has since gone up and is now regularly polling above the threshold.


```
election.plot_pollster_house_effects(samples['election21_2019_polls_pollster_house_effects_b'], hebrew=False)
```

    /usr/local/lib/python3.6/dist-packages/matplotlib/legend.py:508: UserWarning: Automatic legend placement (loc="best") not implemented for figure legend. Falling back on "upper right".
      warnings.warn('Automatic legend placement (loc="best") not '
    


![png](/images/2019-03-17-Forecasting-the-Israeli-Elections-using-pymc3/2019-03-17-Forecasting-the-Israeli-Elections-using-pymc3_15_1.png)



![png](/images/2019-03-17-Forecasting-the-Israeli-Elections-using-pymc3/2019-03-17-Forecasting-the-Israeli-Elections-using-pymc3_15_2.png)



![png](/images/2019-03-17-Forecasting-the-Israeli-Elections-using-pymc3/2019-03-17-Forecasting-the-Israeli-Elections-using-pymc3_15_3.png)



![png](/images/2019-03-17-Forecasting-the-Israeli-Elections-using-pymc3/2019-03-17-Forecasting-the-Israeli-Elections-using-pymc3_15_4.png)



![png](/images/2019-03-17-Forecasting-the-Israeli-Elections-using-pymc3/2019-03-17-Forecasting-the-Israeli-Elections-using-pymc3_15_5.png)



![png](/images/2019-03-17-Forecasting-the-Israeli-Elections-using-pymc3/2019-03-17-Forecasting-the-Israeli-Elections-using-pymc3_15_6.png)



![png](/images/2019-03-17-Forecasting-the-Israeli-Elections-using-pymc3/2019-03-17-Forecasting-the-Israeli-Elections-using-pymc3_15_7.png)



![png](/images/2019-03-17-Forecasting-the-Israeli-Elections-using-pymc3/2019-03-17-Forecasting-the-Israeli-Elections-using-pymc3_15_8.png)



![png](/images/2019-03-17-Forecasting-the-Israeli-Elections-using-pymc3/2019-03-17-Forecasting-the-Israeli-Elections-using-pymc3_15_9.png)



![png](/images/2019-03-17-Forecasting-the-Israeli-Elections-using-pymc3/2019-03-17-Forecasting-the-Israeli-Elections-using-pymc3_15_10.png)



![png](/images/2019-03-17-Forecasting-the-Israeli-Elections-using-pymc3/2019-03-17-Forecasting-the-Israeli-Elections-using-pymc3_15_11.png)



![png](/images/2019-03-17-Forecasting-the-Israeli-Elections-using-pymc3/2019-03-17-Forecasting-the-Israeli-Elections-using-pymc3_15_12.png)



![png](/images/2019-03-17-Forecasting-the-Israeli-Elections-using-pymc3/2019-03-17-Forecasting-the-Israeli-Elections-using-pymc3_15_13.png)



![png](/images/2019-03-17-Forecasting-the-Israeli-Elections-using-pymc3/2019-03-17-Forecasting-the-Israeli-Elections-using-pymc3_15_14.png)



![png](/images/2019-03-17-Forecasting-the-Israeli-Elections-using-pymc3/2019-03-17-Forecasting-the-Israeli-Elections-using-pymc3_15_15.png)



![png](/images/2019-03-17-Forecasting-the-Israeli-Elections-using-pymc3/2019-03-17-Forecasting-the-Israeli-Elections-using-pymc3_15_16.png)


### Correlation Matrix

Finally, we can plot our correlation distributions. Some of these make sense, while others do not. The model tries to make the best of what it can determine from the polls. With 16 parties competing, the model might make some false associations due to parties going up or down in polls the same day. It is a best effort, but in my opinion, better than simply assuming the respective party supports are totally unrelated.


```
from pyhoshen import utils
correlation_matrices = utils.compute_correlations(samples['election21_2019_cholesky_matrix'])
election.plot_election_correlation_matrices(correlation_matrices, hebrew=False)
election.plot_election_correlation_matrices(correlation_matrices, hebrew=True)
```


![png](/images/2019-03-17-Forecasting-the-Israeli-Elections-using-pymc3/2019-03-17-Forecasting-the-Israeli-Elections-using-pymc3_17_0.png)



![png](/images/2019-03-17-Forecasting-the-Israeli-Elections-using-pymc3/2019-03-17-Forecasting-the-Israeli-Elections-using-pymc3_17_1.png)

