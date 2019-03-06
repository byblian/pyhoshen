# coding: utf-8
"""
A collection of pymc3 models for political election modeling.
"""

import numpy as np
import pymc3 as pm
import theano.tensor as T
import datetime
import itertools

from . import polls
from . import configuration

def get_version():
    return 365

class ElectionDynamicsModel(pm.Model):
    """
    A pymc3 model that models the dynamics of an election
    campaign based on the polls, optionally assuming "house
    effects."
    """
    def __init__(self, name, votes, polls, party_groups,
                 cholesky_matrix, test_results, model_type):
        super(ElectionDynamicsModel, self).__init__(name)
        
        self.votes = votes
        self.polls = polls
        self.party_groups = party_groups
        
        self.num_parties = polls.num_parties
        self.num_days = polls.num_days
        self.num_pollsters = polls.num_pollsters
        self.max_poll_days = polls.max_poll_days
        self.num_party_groups = max(self.party_groups) + 1
        self.cholesky_matrix = cholesky_matrix
        
        self.test_results = (polls.get_last_days_average(10)
            if test_results is None else test_results)

        # The base polls model. House-effects models
        # are optionally set up based on this model.

        # The innovations are multivariate normal with the same
        # covariance/cholesky matrix as the polls' MvStudentT
        # variable. The assumption is that the parties' covariance
        # is invariant throughout the election campaign and
        # influences polls, evolving support and election day
        # vote.
        self.innovations = pm.MvNormal('innovations',
            mu=np.zeros([self.num_days, self.num_parties]),
            chol=self.cholesky_matrix,
            shape=[self.num_days, self.num_parties],
            testval=np.zeros([self.num_days, self.num_parties]))
            
        # The random walk itself is a cumulative sum of the innovations.
        self.walk = pm.Deterministic('walk', self.innovations.cumsum(axis=0))

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
            itertools.groupby(sorted(self.polls, key=group_polls), group_polls) ]
            
        # To handle multiple-day polls, we average the party support for the
        # relevant days
        def expected_poll_outcome(p):
            if p.num_poll_days > 1:
                poll_days = [ d for d in range(p.end_day, p.start_day + 1)]
                return self.walk[poll_days].mean(axis=0)
            else:
                return self.walk[p.start_day]
        
        mus = { num_poll_days: [ expected_poll_outcome(p) for p in polls ] + self.votes
                for num_poll_days, polls in self.grouped_polls }

        self.mus = self.create_house_effects(model_type, mus)

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
                mu=mus[num_poll_days],
                chol=self.cholesky_matrix / np.sqrt(num_poll_days),
                testval=test_results,
                shape=[len(polls), self.num_parties],
                observed=[ p.percentages for p in polls ])
            for num_poll_days, polls in self.grouped_polls ]
        
    def create_house_effects(self, model_type, mus, pollster_sigma_beta = 0.05):
        # Create the appropriate house-effects model, if needed.
        if model_type == 'polls-only':
            return mus
        elif model_type == 'mult-mean-variance':
            # Model the coefficient multiplied on the mean as
            # a Gamma variable per-pollster per-party

            self.pollster_house_effects = pm.Gamma(
                'pollster_house_effects', 1, 1,
                shape=[self.num_pollsters, self.num_parties])
            self.pollster_house_effects_eps = pm.HalfCauchy('pollster_house_effects_eps', 0.05)
            self.pollster_house_effects_avg = pm.Potential('pollster_house_effects_avg',
                pm.Normal.dist(1, self.pollster_house_effects_eps, shape=self.num_pollsters).logp(
                    self.pollster_house_effects.prod(axis=1)))

            # Model the variance of the pollsters as a HalfCauchy
            # variable.
            self.pollster_sigmas = pm.HalfCauchy('pollster_sigmas',
                pollster_sigma_beta, shape=[self.num_pollsters, 1])
    
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
            def create_mult_mean_variance_mu(num_poll_days, polls):
                pollster_ids = [ p.pollster_id for p in polls ]
                offsets = pm.Normal(
                    'offsets_%d' % num_poll_days,
                    0, 1, shape=[len(polls), 1])
                
                return (self.pollster_house_effects[pollster_ids] * mus[num_poll_days]
                    + self.pollster_sigmas[pollster_ids] * offsets)
                
            return { num_poll_days: create_mult_mean_variance_mu(num_poll_days, polls)
                     for num_poll_days, polls in self.grouped_polls }

        elif model_type == 'variance':
            self.pollster_house_effects = pm.Deterministic(
                'pollster_house_effects', 
                T.ones([self.num_pollsters, self.num_parties]))

            # Model the variance of the pollsters as a HalfCauchy
            # variable.
            self.pollster_sigmas = pm.HalfCauchy('pollster_sigmas',
                pollster_sigma_beta, shape=[self.num_pollsters, 1])
    
            def create_variance_mu(num_poll_days, polls):
                pollster_ids = [ p.pollster_id for p in polls ]
                offsets = pm.Normal(
                    'offsets_%d' % num_poll_days,
                    0, 1, shape=[len(polls), 1])
                
                return (mus[num_poll_days] + self.pollster_sigmas[pollster_ids] * offsets)
                
            return { num_poll_days: create_mult_mean_variance_mu(num_poll_days, polls)
                     for num_poll_days, polls in self.grouped_polls }

        elif model_type == 'party-variance':
            self.pollster_house_effects = pm.Deterministic(
                'pollster_house_effects', 
                T.ones([self.num_pollsters, self.num_parties]))

            # Model the variance of the pollsters as a HalfCauchy
            # variable.
            self.pollster_sigmas = pm.HalfCauchy('pollster_sigmas',
                pollster_sigma_beta, shape=[self.num_pollsters, self.num_parties])
    
            def create_party_variance_mu(num_poll_days, polls):
                pollster_ids = [ p.pollster_id for p in polls ]
                offsets = pm.Normal(
                    'offsets_%d' % num_poll_days,
                    0, 1, shape=[len(polls), self.num_parties])
                
                return (mus[num_poll_days] + self.pollster_sigmas[pollster_ids] * offsets)
                
            return { num_poll_days: create_mult_mean_variance_mu(num_poll_days, polls)
                     for num_poll_days, polls in self.grouped_polls }

        else:
            raise ValueError("expected model_type '%s' to be one of %s" % 
                (model_type, ', '.join(['polls-only', 
                                        'mult-mean-variance',
                                        'variance',
                                        'party-variance'])))
        
        
class ElectionCycleModel(pm.Model):
    """
    A pymc3 model that models the full election cycle. This can
    include a fundamentals model as well as a dynamics model. 
    """
    def __init__(self, election_model, name, cycle_config, parties, election_polls,
                 eta, test_results=None, real_results=None, model_type=None):
        super(ElectionCycleModel, self).__init__(name)

        self.config = cycle_config

        self.forecast_day = election_polls.forecast_day
        self.election_polls = election_polls
        self.parties = parties
        self.num_days = election_polls.num_days
        self.pollster_ids = election_polls.pollster_ids
        self.party_ids = election_polls.party_ids
        
        self.num_parties = len(self.parties)
        self.eta = eta
        
        # Create the cholesky matrix of the model
        self.cholesky_pmatrix = pm.LKJCholeskyCov('cholesky_pmatrix',
            n=self.num_parties, eta=self.eta,   
            sd_dist=pm.HalfCauchy.dist(0.1, shape=[self.num_parties]))
        self.cholesky_matrix = pm.Deterministic('cholesky_matrix',
            pm.expand_packed_triangular(self.num_parties, self.cholesky_pmatrix))
        
        # Model the prior on the election-day votes
        # This could be replaced by the results of a
        # fundamentals model
        self.votes = pm.Uniform('votes', 0, 0.5, shape=self.num_parties)

        # Prepare the party grouping indexes. This is
        # currently unused.
        self.groups = []
        self.party_groups = []
        for p in self.party_ids:
            group = self.parties[p]['group']
            if group not in self.groups:
                self.groups += [ group ]
            self.party_groups += [ self.groups.index(group) ]

        # Create the Dynamics model.
        self.dynamics = ElectionDynamicsModel(
            name=name + '_polls', votes=self.votes, 
            polls=election_polls, party_groups=self.party_groups,
            cholesky_matrix=self.cholesky_matrix,
            test_results=test_results, model_type=model_type)
            
        self.support = pm.Deterministic('support', self.dynamics.support)

class ElectionForecastModel(pm.Model):
    """
    A pymc3 model that models the election forecast, based on
    one or more election cycles.
    """
    def __init__(self, config, forecast_election=None,
                 base_elections=None, forecast_day=None,
                 eta=25, model_type='polls', 
                 max_days=35, *args, **kwargs):

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
            forecast_day=forecast_day, real_results=None, max_days=max_days,
            eta=eta, model_type=model_type)
               
        self.support = pm.Deterministic('support', self.forecast_model.support)

    def init_cycle(self, cycle, forecast_day, real_results, 
                   eta, model_type, max_days = None):                
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
        
        # Use the election day if the forecast day was not provided
        if forecast_day is None:
            forecast_day = datetime.datetime.strptime(cycle_config['election_day'], '%d/%m/%Y').date()

        # Multiple poll datasets are not yet supported at this
        # interface so use the first dataset ('-0')
        election_polls = polls.ElectionPolls(
            self.config.dataframes['polls']['%s-0' % cycle],
            parties.keys(), forecast_day, max_days)
        
        test_results = [ np.nan_to_num(f) for f in election_polls.get_last_days_average(10)]

        return ElectionCycleModel(self, cycle, cycle_config, parties,
            election_polls=election_polls, eta=eta,
            test_results=test_results, real_results=real_results, model_type=model_type)
