import numpy as np

def get_version():
    return 7

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
    
    def __init__(self, polls_dataset, party_ids, forecast_day, max_days = None):

        def day_index(d):
            return (forecast_day - d.to_pydatetime().date()).days
    
        self.forecast_day = forecast_day
        self.party_ids = [p for p in party_ids]
        self.num_parties = len(self.party_ids)
        self.num_days = day_index(min(polls_dataset['start_date'])) + 1
        if max_days is not None:
            self.num_days = min(self.num_days, max_days)
            
        self.max_poll_days = 0

        self.pollster_ids = []
        self.polls = []

        missing_parties = [p for p in self.party_ids if p not in polls_dataset.columns]
        assert len(missing_parties) == 0, "parties %s are missing for %s" % (str(missing_parties), str(forecast_day))
        for index, poll in polls_dataset.iterrows():
            percentages = poll[self.party_ids]
            pollster = poll['pollster'] if 'pollster' in poll else poll['poller']
            poll_id = poll['id']
            num_polled = poll['num_polled']
            num_poll_days = poll['num_days']
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
