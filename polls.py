import numpy as np
import pandas as pd
import datetime as dt

class Poll:
    def __init__(self, poll_id, num_polled, start_day, num_poll_days, percentages, pollster_id):
        assert num_polled >= 100, "expected num_polled >= 100, but was %d" % num_polled
        self.poll_id = poll_id
        self.num_polled = num_polled
        self.start_day = start_day
        self.end_day = start_day - num_poll_days + 1
        self.num_poll_days = num_poll_days
        self.percentages = percentages
        self.pollster_id = pollster_id
       
class ElectionPolls:
    
    def __init__(self, polls_dataset, party_ids, forecast_day,
                 extra_avg_days=0, max_poll_days=None, polls_since=None, min_poll_days=None):

        def day_index(d):
            if type(d) is pd.Timestamp:
                d = d.to_pydatetime()
            if type(d) is dt.datetime:
                d = d.date()
            assert type(d) == dt.date, "invalid value given for date: %s" % str(d)
            return (forecast_day - d).days + (extra_avg_days + 1) // 2
    
        self.forecast_day = forecast_day
        self.party_ids = [p for p in party_ids]
        self.num_parties = len(self.party_ids)
        
        self.num_days = day_index(min(polls_dataset['start_date'])) + 1
        if max_poll_days is not None:
            assert polls_since==None, "only one of polls_since or max_poll_days should be provided"
            self.num_days = min(self.num_days, max_poll_days)
        elif polls_since is not None:
            polls_since_days = max(day_index(polls_since) + 1, min_poll_days)
            self.num_days = min(self.num_days, polls_since_days)
        elif self.num_days > 90:
            print ("poll days truncated to 90, originally ", self.num_days, " forecast day:", forecast_day)
            self.num_days = min(self.num_days, 90)
        self.max_poll_days = 0

        self.pollster_ids = []
        self.polls = []

        missing_parties = [p for p in self.party_ids if p not in polls_dataset.columns]
        assert len(missing_parties) == 0, "parties %s are missing for %s" % (str(missing_parties), str(forecast_day))
        for index, poll in polls_dataset.iterrows():
            percentages = poll[self.party_ids]
            pollster = poll['pollster'] if 'pollster' in poll else poll['poller']
            poll_id = index
            num_polled = poll['num_polled']
            num_poll_days = poll['num_days'] + extra_avg_days
            start_day = day_index(poll['start_date'])

            if start_day - num_poll_days + 1 >= 0 and start_day < self.num_days:
                assert len(percentages) == self.num_parties, "percentages list does not match"
                if pollster not in self.pollster_ids:
                    self.pollster_ids += [ pollster ]
                poll_id = len(self.polls)
                self.polls += [ Poll(poll_id, num_polled, start_day, num_poll_days,
                                     percentages, self.pollster_ids.index(pollster))]
                self.max_poll_days = max(self.max_poll_days, num_poll_days)

        self.num_pollsters = len(self.pollster_ids)
    
    def get_last_days_average(self, num_days):
        return np.stack([ 
                p.percentages
                for p in self.polls 
                if p.end_day < num_days ], axis=0).mean(axis=0)
    
    def __iter__(self):
        return self.polls.__iter__()
    
    def __len__(self):
        return len(self.polls)

