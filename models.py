# coding: utf-8
"""
A collection of pymc3 models for political election modeling.
"""

import numpy as np
import pymc3 as pm
import theano.tensor as tt
import datetime
import itertools

from . import polls
from . import configuration

class ElectionDynamicsModel(pm.Model):
    """
    A pymc3 model that models the dynamics of an election
    campaign based on the polls, optionally assuming "house
    effects."
    """
    def __init__(self, name, votes, polls, cholesky_matrix,
                 after_polls_cholesky_matrix, election_day_cholesky_matrix,
                 test_results, house_effects_model, min_polls_per_pollster,
                 adjacent_day_fn):
        super(ElectionDynamicsModel, self).__init__(name)
        
        self.votes = votes
        self.polls = polls
        
        self.num_parties = polls.num_parties
        self.num_days = polls.num_days
        self.num_pollsters = polls.num_pollsters
        self.max_poll_days = polls.max_poll_days

        self.cholesky_matrix = cholesky_matrix
        self.after_polls_cholesky_matrix = after_polls_cholesky_matrix
        self.election_day_cholesky_matrix = election_day_cholesky_matrix

        if type(adjacent_day_fn) in [int, float]:
            self.adjacent_day_fn = lambda diff: (1. + diff) ** adjacent_day_fn
        else:
            self.adjacent_day_fn = adjacent_day_fn
        
        self.test_results = (polls.get_last_days_average(10)
            if test_results is None else test_results)
        
        # In some cases, we might want to filter pollsters without a minimum
        # number of polls. Because these pollsters produced only a few polls,
        # we cannot determine whether their results are biased or not.
        polls_per_pollster = { 
            pollster_id: sum(1 for p in self.polls if p.pollster_id == pollster_id) 
            for pollster_id in range(self.num_pollsters) }
        
        self.min_polls_per_pollster = min_polls_per_pollster
        
        self.num_pollsters_in_model = 0
        self.pollster_mapping = {}
    
        for pollster_id, count in polls_per_pollster.items():
            if count >= self.min_polls_per_pollster:
                self.pollster_mapping[pollster_id] = self.num_pollsters_in_model
                self.num_pollsters_in_model += 1
            else:
                self.pollster_mapping[pollster_id] = None
        
        self.filtered_polls = [ p for p in self.polls 
                               if polls_per_pollster[p.pollster_id] >= self.min_polls_per_pollster ]
        
        if len(self.polls) - len(self.filtered_polls) > 0:
          print ("Some polls were filtered out. Provided polls: %d, filtered: %d, final total: %d" % 
             (len(self.polls), len(self.polls) - len(self.filtered_polls), len(self.filtered_polls)))
        else:
          print ("Using all %d provided polls." % len(self.polls))

        self.first_poll_day =  min(p.end_day for p in self.filtered_polls)
        
        # The base polls model. House-effects models
        # are optionally set up based on this model.

        # The innovations are multivariate normal with the same
        # covariance/cholesky matrix as the polls' MvStudentT
        # variable. The assumption is that the parties' covariance
        # is invariant throughout the election campaign and
        # influences polls, evolving support and election day
        # vote.
        self.innovations = [ pm.MvNormal('election_day_innovations',
            mu=np.zeros([1, self.num_parties]),
            chol=self.election_day_cholesky_matrix,
            shape=[1, self.num_parties],
            testval=np.zeros([1, self.num_parties])) ]

        if self.first_poll_day > 1:
            self.innovations += [ pm.MvNormal('after_poll_innovations',
                mu=np.zeros([self.first_poll_day - 1, self.num_parties]),
                chol=self.after_polls_cholesky_matrix,
                shape=[self.first_poll_day - 1, self.num_parties],
                testval=np.zeros([self.first_poll_day - 1, self.num_parties])) ]

        self.innovations += [ pm.MvNormal('poll_innovations',
            mu=np.zeros([self.num_days - max(self.first_poll_day, 1), self.num_parties]),
            chol=self.cholesky_matrix,
            shape=[self.num_days - max(self.first_poll_day, 1), self.num_parties],
            testval=np.zeros([self.num_days - max(self.first_poll_day, 1), self.num_parties])) ]



        # The random walk itself is a cumulative sum of the innovations.
        self.walk = pm.Deterministic('walk', tt.concatenate(self.innovations, axis=0).cumsum(axis=0))

        # The modeled support of the various parties over time is the sum
        # of both the election-day votes and the innovations that led up to it.
        # The support at day 0 is the election day vote.
        self.support = pm.Deterministic('support', self.votes + self.walk)
        
        # Group polls by number of days. This is necessary to allow generating
        # a different cholesky matrix for each. This corresponds to the 
        # average of the modeled support used for multi-day polls.
        group_polls = lambda poll: poll.num_poll_days

        # Group the polls and create the likelihood variable.
        self.grouped_polls = [ (num_poll_days, [p for p in polls]) for num_poll_days, polls in
            itertools.groupby(sorted(self.filtered_polls, key=group_polls), group_polls) ]
            
        # To handle multiple-day polls, we average the party support for the
        # relevant days
        def expected_poll_outcome(p):
            if p.num_poll_days > 1:
                poll_days = [ d for d in range(p.end_day, p.start_day + 1)]
                return self.walk[poll_days].mean(axis=0)
            else:
                return self.walk[p.start_day]
              
        def expected_polls_outcome(polls):
            if self.adjacent_day_fn is None:
                return [ expected_poll_outcome(p) for p in polls ] + self.votes
            else:
                weights = np.asarray([[ 
                    sum(self.adjacent_day_fn(abs(d - poll_day)) 
                        for poll_day in range(p.end_day, p.start_day + 1)) if d >= self.first_poll_day else 0
                    for d in range(self.num_days) ]
                    for p in polls])
                return tt.dot(weights / weights.sum(axis=1, keepdims=True), self.walk + self.votes)
        
        self.mus = { num_poll_days: expected_polls_outcome(polls)
                for num_poll_days, polls in self.grouped_polls }

        self.create_house_effects(house_effects_model)

        self.likelihoods = [
            # The Multivariate Student-T variable that models the polls.
            #
            # The polls are modeled as a MvStudentT distribution which allows to
            # take into consideration the number of people polled as well as the
            # cholesky-covariance matrix that is central to the model.

            # Because we average the support over the number of poll days n, we
            # also need to appropriately factor the cholesky matrix. We assume
            # no correlation between different days, so the factor is 1/n for 
            # the variance, and 1/sqrt(n) for the cholesky matrix.
            pm.MvStudentT(
                'polls_%d_days' % num_poll_days,
                nu=[ p.num_polled - 1 for p in polls ],
                mu=self.mus[num_poll_days],
                chol=self.cholesky_matrix / np.sqrt(num_poll_days),
                testval=test_results,
                shape=[len(polls), self.num_parties],
                observed=[ p.percentages for p in polls ])
            for num_poll_days, polls in self.grouped_polls ]
        
    def create_house_effects(self, house_effects_model, pollster_sigma_beta = 0.05):
        # Create the appropriate house-effects model, if needed.
        if house_effects_model == 'raw-polls':
            return self.mus

        elif house_effects_model in [ 'add-mean', 'add-mean-variance', 'mult-mean', 'mult-mean-variance', 'lin-mean', 'lin-mean-variance' ]:
            if house_effects_model in [ 'mult-mean', 'mult-mean-variance', 'lin-mean', 'lin-mean-variance' ]:
                # Model the coefficient multiplied on the mean as
                # a Gamma variable per-pollster per-party
                self.pollster_house_effects_a_ = pm.Gamma(
                    'pollster_house_effects_a_', 1, 0.05,
                    shape=[self.num_pollsters_in_model - 1, self.num_parties],
                    testval=tt.ones([self.num_pollsters_in_model - 1, self.num_parties]))
                self.pollster_house_effects_a = pm.Deterministic(
                    'pollster_house_effects_a', 
                    tt.concatenate([self.pollster_house_effects_a_, 
                                    self.num_pollsters_in_model - self.pollster_house_effects_a_.sum(axis=0, keepdims=True)]))
            else:
                self.pollster_house_effects_a = pm.Deterministic(
                    'pollster_house_effects_a', tt.ones([self.num_pollsters_in_model, self.num_parties]))
                
            
            if house_effects_model in [ 'add-mean', 'add-mean-variance', 'lin-mean', 'lin-mean-variance' ]:
                self.pollster_house_effects_b__ = pm.Normal(
                    'pollster_house_effects_b__', 0, 0.05,
                    shape=[self.num_pollsters_in_model - 1, self.num_parties - 1],
                    testval=tt.zeros([self.num_pollsters_in_model - 1, self.num_parties - 1]))
                self.pollster_house_effects_b_ = pm.Deterministic(
                    'pollster_house_effects_b_', 
                    tt.concatenate([self.pollster_house_effects_b__, -self.pollster_house_effects_b__.sum(axis=1, keepdims=True)], axis=1))
                self.pollster_house_effects_b = pm.Deterministic(
                    'pollster_house_effects_b', 
                    tt.concatenate([self.pollster_house_effects_b_, -self.pollster_house_effects_b_.sum(axis=0, keepdims=True)]))
            else:
                self.pollster_house_effects_b= pm.Deterministic(
                    'pollster_house_effects_b', tt.zeros([self.num_pollsters_in_model, self.num_parties]))
                    
           # Model the variance of the pollsters as a HalfCauchy
            # variable.
            if house_effects_model in [ 'add-mean-variance', 'mult-mean-variance', 'lin-mean-variance' ]:
                self.pollster_sigmas = pm.HalfCauchy('pollster_sigmas',
                    pollster_sigma_beta, shape=[self.num_pollsters_in_model, 1])
            else:
                self.pollster_sigmas = pm.Deterministic('pollster_sigmas', 
                    tt.zeros([self.num_pollsters_in_model, 1]))
    
            # To simplify the modeling, only the mean is modified
            # based on the house effects.
            #
            # It is modeled as:
            #   mu ~ N(c_jk * orig_mu, s_j^2)
            #   s_j ~ HC(pollster_sigma_beta)
            #
            # for
            #   j = pollster_id
            #   k = party_id
            #
            # This is transformed to a non-centered parameterization.
            # Because only the mean is modified, the same grouping
            # as the base model can still be used.
            def create_lin_mean_variance_mu(num_poll_days, polls):
                pollster_ids = [ self.pollster_mapping[p.pollster_id] for p in polls ]

                if house_effects_model in [ 'add-mean-variance', 'mult-mean-variance', 'lin-mean-variance' ]:
                    offsets = pm.Normal(
                        'offsets_%d' % num_poll_days,
                        0, 1, shape=[len(polls), 1],
                        testval=np.zeros([len(polls), 1]))
                else:
                    offsets = pm.Deterministic('offsets_%d' % num_poll_days, 
                        tt.zeros([len(polls), 1]))
                
                return (self.pollster_house_effects_a[pollster_ids] * self.mus[num_poll_days] + 
                        self.pollster_house_effects_b[pollster_ids] +
                        self.pollster_sigmas[pollster_ids] * offsets)
              
            self.mus = { num_poll_days: create_lin_mean_variance_mu(num_poll_days, polls)
                     for num_poll_days, polls in self.grouped_polls }
            
        elif house_effects_model == 'variance':
            self.pollster_house_effects = pm.Deterministic(
                'pollster_house_effects', 
                tt.ones([self.num_pollsters, self.num_parties]))

            # Model the variance of the pollsters as a HalfCauchy
            # variable.
            self.pollster_sigmas = pm.HalfCauchy('pollster_sigmas',
                pollster_sigma_beta, shape=[self.num_pollsters, 1])
    
            def create_variance_mu(num_poll_days, polls):
                pollster_ids = [ p.pollster_id for p in polls ]
                offsets = pm.Normal(
                    'offsets_%d' % num_poll_days,
                    0, 1, shape=[len(polls), 1],
                    testval=np.zeros([len(polls), 1]))
                
                return (self.mus[num_poll_days] + self.pollster_sigmas[pollster_ids] * offsets)
                
            self.mus = { num_poll_days: create_variance_mu(num_poll_days, polls)
                     for num_poll_days, polls in self.grouped_polls }

        elif house_effects_model == 'party-variance':
            self.pollster_house_effects = pm.Deterministic(
                'pollster_house_effects', 
                tt.ones([self.num_pollsters, self.num_parties]))

            # Model the variance of the pollsters as a HalfCauchy
            # variable.
            self.pollster_sigmas = pm.HalfCauchy('pollster_sigmas',
                pollster_sigma_beta, shape=[self.num_pollsters, self.num_parties])
    
            def create_party_variance_mu(num_poll_days, polls):
                pollster_ids = [ p.pollster_id for p in polls ]
                offsets = pm.Normal(
                    'offsets_%d' % num_poll_days,
                    0, 1, shape=[len(polls), self.num_parties],
                    testval=np.zeros([len(polls), self.num_parties]))
                
                return (self.mus[num_poll_days] + self.pollster_sigmas[pollster_ids] * offsets)
                
            self.mus = { num_poll_days: create_party_variance_mu(num_poll_days, polls)
                     for num_poll_days, polls in self.grouped_polls }

        else:
            raise ValueError("expected model_type '%s' to be one of %s" % 
                (house_effects_model, ', '.join(['raw-polls', 
                                                 'add-mean',
                                                 'add-mean-variance',
                                                 'mult-mean-variance',
                                                 'lin-mean-variance',
                                                 'variance',
                                                 'party-variance'])))
        
        
class ElectionCycleModel(pm.Model):
    """
    A pymc3 model that models the full election cycle. This can
    include a fundamentals model as well as a dynamics model. 
    """
    def __init__(self, election_model, name, cycle_config, parties, election_polls,
                 eta, adjacent_day_fn, min_polls_per_pollster,
                 test_results=None, real_results=None,
                 house_effects_model=None, chol=None, votes=None,
                 after_polls_chol=None, election_day_chol=None):
        super(ElectionCycleModel, self).__init__(name)

        self.config = cycle_config

        self.house_effects_model = house_effects_model
        self.forecast_day = election_polls.forecast_day
        self.election_polls = election_polls
        self.parties = parties
        self.num_days = election_polls.num_days
        self.pollster_ids = election_polls.pollster_ids
        self.party_ids = election_polls.party_ids
        
        self.num_parties = len(self.parties)
        self.eta = eta
        
        # Create the cholesky matrix of the model
        if chol is None:
          self.cholesky_pmatrix = pm.LKJCholeskyCov('cholesky_pmatrix',
              n=self.num_parties, eta=self.eta,   
              sd_dist=pm.HalfCauchy.dist(0.1, shape=[self.num_parties]))
          self.cholesky_matrix = pm.Deterministic('cholesky_matrix',
              pm.expand_packed_triangular(self.num_parties, self.cholesky_pmatrix))
        else:
          self.cholesky_matrix = chol

        self.after_polls_cholesky_matrix=after_polls_chol if after_polls_chol is not None else self.cholesky_matrix
        self.election_day_cholesky_matrix=election_day_chol if election_day_chol is not None else self.cholesky_matrix 
        
        # Model the prior on the election-day votes
        # This could be replaced by the results of a
        # fundamentals model
        if votes is None:
          self.votes = pm.Flat('votes', shape=self.num_parties)
        else:
          self.votes = votes

        # Prepare the party grouping indexes. This is
        # currently unused.

        # Create the Dynamics model.
        self.dynamics = ElectionDynamicsModel(
            name=name + '_polls', votes=self.votes, 
            polls=election_polls, cholesky_matrix=self.cholesky_matrix,
            after_polls_cholesky_matrix=self.after_polls_cholesky_matrix,
            election_day_cholesky_matrix=self.election_day_cholesky_matrix,
            test_results=test_results, house_effects_model=house_effects_model,
            min_polls_per_pollster=min_polls_per_pollster,
            adjacent_day_fn=adjacent_day_fn)
            
        self.support = pm.Deterministic('support', self.dynamics.support)

class ElectionForecastModel(pm.Model):
    """
    A pymc3 model that models the election forecast, based on
    one or more election cycles.
    """
    def __init__(self, config, forecast_election=None,
                 base_elections=None, forecast_day=None,
                 eta=1, min_polls_per_pollster=1,
                 house_effects_model='add-mean', 
                 extra_avg_days=0, max_poll_days=None, 
                 polls_since=None, min_poll_days=None,
                 adjacent_day_fn=-2., join_composites=False,
                 votes=None, chol=None, after_polls_chol=None,
                 election_day_chol=None, allow_after_election_day=False,
                 *args, **kwargs):

        super(ElectionForecastModel, self).__init__(*args, **kwargs)
        
        self.config = configuration.Configuration(config)

        if forecast_election is None:
            forecast_election = max(self.config['cycles'])
        
        self.forecast_election = forecast_election

        # Base elections can be used to forecast based on
        # the results of the model based on historical
        # data. Currently not implemented.
        if base_elections is None:
            base_elections = [ cycle for cycle in self.config['cycles']
                if cycle < forecast_election]

        self.forecast_model = self.init_cycle(forecast_election, 
            forecast_day=forecast_day, real_results=None,
            extra_avg_days=extra_avg_days, max_poll_days=max_poll_days,
            polls_since=polls_since, min_poll_days=min_poll_days,
            eta=eta, house_effects_model=house_effects_model,
            min_polls_per_pollster=min_polls_per_pollster,
            adjacent_day_fn=adjacent_day_fn, join_composites=join_composites,
            votes=votes, chol=chol, after_polls_chol=after_polls_chol,
            election_day_chol=election_day_chol,
            allow_after_election_day=allow_after_election_day)
               
        self.support = pm.Deterministic('support', self.forecast_model.support)

    def init_cycle(self, cycle, forecast_day, real_results, 
                   eta, min_polls_per_pollster, house_effects_model,
                   extra_avg_days, max_poll_days, polls_since, min_poll_days,
                   adjacent_day_fn, join_composites,
                   votes, chol, after_polls_chol, election_day_chol,
                   allow_after_election_day):
        cycle_config = self.config['cycles'][cycle]

        parties = cycle_config['parties']

        # Remove any parties that are no longer participating due to unions
        # (Percentages in polls will be accounted for separately)
        party_unions = { p: parties[p]['union_of'] for p in parties if 'union_of' in parties[p] }
        for composite, components in party_unions.items():
            for c in components:
                if c != composite and c in parties:
                    del parties[c]

        # Remove any parties that are no longer participating due to dissolutions
        for p in [ id for id, party in parties.items() if 'dissolved' in party ]:
            del parties[p]
        
        # Read the polls
        # Note that only the first polls series will actually be used
        for i, poll_config in enumerate(cycle_config['polls']):
            self.config.read_polls(cycle_config, {'%s-%d' % (cycle, i): poll_config})
        
        # Use the election day if the forecast day was not provided or is later than election day
        # (to allow reasonable use of forecast today + 5 days)
        election_day = datetime.datetime.strptime(cycle_config['election_day'], '%d/%m/%Y').date()
        if forecast_day is None:
          forecast_day = election_day
        elif not allow_after_election_day:
          forecast_day = min(forecast_day, election_day)

        # Multiple poll datasets are not yet supported at this
        # interface so use the first dataset ('-0')
        election_polls = polls.ElectionPolls(
            self.config.dataframes['polls']['%s-0' % cycle],
            parties.keys(), forecast_day, extra_avg_days,
            max_poll_days, polls_since, min_poll_days)
        
        test_results = [ np.nan_to_num(f) for f in election_polls.get_last_days_average(10)]

        return ElectionCycleModel(self, cycle, cycle_config, parties,
            election_polls=election_polls, eta=eta,
            test_results=test_results, real_results=real_results,
            house_effects_model=house_effects_model,
            min_polls_per_pollster=min_polls_per_pollster,
            adjacent_day_fn=adjacent_day_fn, votes=votes,
            chol=chol, after_polls_chol=after_polls_chol,
            election_day_chol=election_day_chol)
