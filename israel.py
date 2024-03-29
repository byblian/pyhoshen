# -*- coding: utf-8 -*-
"""
Supporting analysis functions of Israeli Election results.
"""

from . import models
from . import utils
import theano
from theano.scan.utils import until as theano_scan_until
import theano.tensor as tt
from theano.ifelse import ifelse
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns
import datetime

def strpdate(d):
    return datetime.datetime.strptime(d, '%d/%m/%Y').date()

class IsraeliElectionForecastModel(models.ElectionForecastModel):
    """
    A class that encapsulates computations specific to the Israeli Election
    such as Bader-Ofer Knesset seat computations.
    """
    def __init__(self, config, *args, **kwargs):
        super(IsraeliElectionForecastModel, self).__init__(config, *args, **kwargs)
        
        self.generated_by = 'Generated using pyHoshen © 2019 - 2022\n'

    def create_logo(self):
      from PIL import Image, ImageDraw, ImageFont
      from io import BytesIO
      import base64
    
      # https://www.iconfinder.com/icons/1312097/circle_github_outline_social-media_icon
      github_b64 = 'iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEwAACxMBAJqcGAAAAzFJREFUSImdlc9LY1cYhp/z5WpQb0IEowFpELTmGhAHdFOCy1ZwSl1I+we4nUXBf8KVCN10Ne2yNItMadEZR1wqFhlbN95rGCjWhWAg/sAkJibndDGJxOTGcfru7nnPed7vfueecxVP1NTUVKRcLg8CBIPB88PDw8unrFOPmY7jfGGMWQLmlFKfNXta63+VUpsi8tJ13T8/KWBsbGxURH4UkS+fUiXwWmv9IpvN/vPRAMdxngO/AKEnwhu6NsZ8d3x8vNkxwHGc51rr30TE+kQ4AFrrOxH52vO8t20B9bb8LSL2/4E3ZIy5FJFnruueAEjDqPf8AVxESKVS9PT0tIGCwSCpVAqlHnZZKRUxxvzQeLbgw9cCtG1oKBRiZGSE/v5+LMvCtj/kF4tFisUiQ0NDHBwcUCqVWpd+4zjOtOd576z6ay21VlIfJ5fLsbW1RTgcvgfZtk0+n2dhYYFardapVUvAu0aL5vwmJRIJzs7OMMZwdXVFpVKhUqmQz+cBOD09ZWJiwjdAKTUHIJOTk/2th6ihWCzGycmJL6ARMDw83MkeTSaTttzd3UU7zbAsCxHpZFOr1R71q9VqtLMLnJ+fMzAw0NGPRqNcXFw8hkCCweC5nxGJRNjb22N2dpZ4PN7mx+Nxpqen2d/f7wi3LCunAMbHx09E5AFlcXGR7u5uNjY2mJmZoa+vj+3tbXp7e1leXiafz7O6uorW2heutX6fzWY/FwCl1GbrhPX1dUSE+fl5rq+vsW2bQqFALpcjFAqxtrbWEV7XG6ifZBF52ere3t6SyWRQSjE4OMju7u69F4vFfE93swKBwE/3AfX7/HXrpFKpRCaTAWBlZeV+XETarohmGWNeua77130AgNb6BXDt9yY7OztorUmn06TTaQqFAuVy2Reutb6wLOv7xvODMhKJxJwx5g8R6WpdGA6HCYfDAFxeXnJzc+MHrwQCgXnXdbd9AwAcx/nKGPOrUiriW2IHaa0vAoHAt81waGpRQ57nvRWRZ8DvT4UbY151dXVNtcLh4z/96fpNOweMtlT8HngjIj97nnfQifFoQLOSyaRdrVaj8OGEHh0dtW+Cj/4DJ6A+XqZUkB4AAAAASUVORK5CYII='
      github_im = Image.open(BytesIO(base64.b64decode(github_b64)))
    
      #https://www.iconfinder.com/icons/1312087/circle_outline_social-media_twitter_icon
      twitter_b64 = 'iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEwAACxMBAJqcGAAAA0VJREFUSImdlU1vU0cUhp8zc20nJCa52A5ISRQJC0VqVYHEChbtphWi0FZp40TqMlJXLPgv3bCirFBFiJNSoZagsuqmXRSli0pZmKYFVQpBsW8hH/66c7rwR+zYJk7P6t55Z573zMyZGaHP+GylMDqglTGAokS2Hsz4QT/j5G3i7L3tS2JlwSlXjDDZqjncc8Gsqrrb2Uzy12MZzC8H6Wrobhnho36yBH7UUG4szfsbRxrMZrevSSjfYoj3CQfAwWuLzi3OJlZ7GtTg+h3GeMeBH5i4Csr1bCb5uMNgfjlIV51bMzD8f+AHLi4IjbmwPHvqbwDTaK+G7lYr/ITXff8FeDdluTzu8V7KwwicGRJGYnWUMaOi+nWjv4FatbRuqD9geH8qgj/QaXJp3KMSwvp2SFByXB63fHIuwmRckHp3I/JpZjG4COABiJWFVkhiUHhWCJlOGN6UYD0fEjo4d8qytafkCiEA+aLyuiQkhywqoNoCMdUF4DcPoF7nzditwNSI4Zd/KpwdtXww6VEOIWqFXODal0wgv6/88Spsa3fKFQD58m7gV2Iu3yomB4WrZz0ePgspFF1z2WJW2dzVNtBE3DA2ZHi6We1YTlwl7oWD5RSuvSqdCpt7UA4PYA2jwxGPCv/20FSjKdNNyBcdv7+s8uFU5O13CeAPGjb3tKduihLZ6iYUQ4gPmGZldIuJuGG/ouyWuxuIlF+ZBzN+4HDPD4tvSsrPL8qcH7NMnrRdBsN0wrC21WXtAYfm7s+d3jEAgmm7PybilrRviRmYTkQ6Bp8ZEq6lozz5q9pemq2hPIL6OVB1t0XMVw0tdUJ4J2n5Mwj5aaNE2dXOxkhMSPuWwr7jYa7cg9wI800t+XpklvI/AFcb/8NRYWJYGB0wWCM4VYKSshE4itXemwrgVFeymcTnzRkAaCg31OqagZMAO2VlPa9A9xLsCccVPE9uNufR+Fia9zcsOudwlWMR2+Fl1GbuzSRedBgALM4mVlGu41xf7+3hzFH7cTbjP2lt7zho2UzycWjMBaf6fd9w1RXPyvnDcDji0c8sBhcx1fqjb9LtGWsO5ZE15s7iF/7TXoyjboIWs5fDqtEU1E7o/bnTO/2M+w/4HErdH/aalgAAAABJRU5ErkJggg=='
      twitter_im = Image.open(BytesIO(base64.b64decode(twitter_b64)))
    
      logo = Image.new('RGBA', (250, 72), None)
      draw = ImageDraw.Draw(logo)
      font = ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',12)
      logo.paste(twitter_im.resize((24,24)), (0,24))
      draw.text((26,29), '@pyHoshen', (0,0,0), font=font)
      logo.paste(github_im.resize((24,24)), (0,48))
      draw.text((26,53), 'https://github.com/byblian/pyhoshen', (0,0,0), font=font)
      draw.text((18,0), 'Generated using pyHoshen © 2019 - 2022', (0,0,0), font=font)
      
      return logo

    def day_index(self, d):
        """
        Computes the days before the forecast date of a given date.
        
        Forecast day is considered day-index 0, and the day index increases
        for each day beforehand.
        """
        return (self.forecast_model.forecast_day - strpdate(d)).days
    
    def house_effects_model_title(self, hebrew = True):
        from bidi import algorithm as bidialg

        fe = self.forecast_model
        
        if fe.house_effects_model == 'raw-polls':
            house_effects_model_name = 'Raw Polls'
        elif fe.house_effects_model == 'add-mean':
            house_effects_model_name = 'Additive Mean'
        elif fe.house_effects_model == 'add-mean-variance':
            house_effects_model_name = 'Additive Mean with Variance'
        elif fe.house_effects_model == 'mult-mean':
            house_effects_model_name = 'Multiplicative Mean'
        elif fe.house_effects_model == 'mult-mean-variance':
            house_effects_model_name = 'Multiplicative Mean with Variance'
        elif fe.house_effects_model == 'lin-mean':
            house_effects_model_name = 'Linear Mean'
        elif fe.house_effects_model == 'lin-mean-variance':
            house_effects_model_name = 'Linear Mean with Variance'
        elif fe.house_effects_model == 'variance':
            house_effects_model_name = 'Variance'
        elif fe.house_effects_model == 'party-variance':
            house_effects_model_name = 'Party-Specific Variance'
        else:
            raise ValueError('Unknown house effects model')
        
        if hebrew:
            return bidialg.get_display('\n(על פי מודל הטיות סוקרים “%s”)' % house_effects_model_name)
        else:
            return '(According to “%s” House-Effects model)' % house_effects_model_name
        
    def create_surplus_matrices(self):
        """
        Create matrices that represent the surplus agreements between
        political parties, used during the Bader-Ofer computations.
        """
        fe = self.forecast_model

        num_parties = len(fe.parties)
        surplus_matrices = np.stack([ np.eye(num_parties, dtype="int64") ] * fe.num_days)
        for day in range(fe.num_days):
            cur_agreements = [ sa for sa in fe.config['surplus_agreements']
                if 'since' not in sa or self.day_index(sa['since']) >= day ]
            for agreement in cur_agreements:
              name1 = agreement['name1']
              name2 = agreement['name2']
              if name1 in fe.party_ids and name2 in fe.party_ids:
                party1 = fe.party_ids.index(name1)
                party2 = fe.party_ids.index(name2)
                surplus_matrices[day, party1, party2] = 1
                surplus_matrices[day, party2, party2] = 0
            
        return surplus_matrices
    
    def compute_trace_bader_ofer(self, trace, surpluses = None, threshold = None):
        """
        Compute the Bader-Ofer on a full sample trace using theano scan.
        
        Example usage:
            bo=election.compute_trace_bader_ofer(samples['support'])
            
        trace should be of dimensions nsamples x ndays x nparties
        """
        # trace : nsamples x ndays x nparties
        num_seats = tt.constant(120)
    
        def bader_ofer_fn___(prior, votes):
            moded = votes / (prior + 1)
            return prior + tt.eq(moded, moded.max())
        
        def bader_ofer_fn__(cur_seats, prior, votes):
            new_seats = ifelse(tt.lt(cur_seats, num_seats), bader_ofer_fn___(prior, votes), prior)
            return (cur_seats + 1, new_seats.astype("int64")), theano_scan_until(tt.ge(cur_seats, num_seats))
        
        # iterate a particular day of a sample, and compute the bader-ofer allocation
        def bader_ofer_fn_(seats, votes, surplus_matrix):
          initial_seats = surplus_matrix.dot(seats)
          comp_ejs__, upd_ejs__ = theano.scan(fn = bader_ofer_fn__,
            outputs_info = [initial_seats.sum(), initial_seats], non_sequences = [surplus_matrix.dot(votes)], n_steps = num_seats)
          joint_seats = comp_ejs__[1][-1]
          surplus_t = surplus_matrix.T
          has_seats_t = surplus_t * tt.gt(surplus_t.sum(0),1)
          is_joint_t = tt.gt(surplus_t.sum(0),1).dot(surplus_matrix)
          non_joint = tt.eq(is_joint_t, 0)
          votes_t = votes.dimshuffle(0, 'x')
          our_votes_t = surplus_t * votes_t
          joint_moded = tt.switch(tt.eq(joint_seats, 0), 0, our_votes_t.sum(0) / joint_seats)
          joint_moded_both_t = joint_moded * has_seats_t
          initial_seats_t = tt.switch(tt.eq(joint_moded_both_t, 0), 0, our_votes_t // joint_moded_both_t)
          moded_t = tt.switch(tt.eq(joint_moded_both_t, 0), 0, votes_t / (initial_seats_t + 1))
          added_seats = tt.eq(moded_t, moded_t.max(0)) * has_seats_t * tt.gt(joint_seats - initial_seats_t, 0)
          joint_added = initial_seats_t.sum(1) + added_seats.sum(1).astype("int64")
          return joint_seats * non_joint + joint_added * is_joint_t * (seats > 0)
        
        # iterate each day of a sample, and compute for each the bader-ofer allocation
        def bader_ofer_fn(seats, votes, surplus_matrices):
          comp_bo_, _ = theano.scan(fn = bader_ofer_fn_, sequences=[seats, votes, surplus_matrices])
          return comp_bo_
        
        if threshold is None:
            threshold = float(self.forecast_model.config['threshold_percent']) / 100

        if surpluses is None:
            surpluses = self.create_surplus_matrices()
            
        votes = tt.tensor3("votes")
        seats = tt.tensor3("seats", dtype='int64')
        surplus_matrices = tt.tensor3("surplus_matrices", dtype='int64')
        
        # iterate each sample, and compute for each the bader-ofer allocation
        comp_bo, _ = theano.scan(bader_ofer_fn, sequences=[seats, votes], non_sequences=[surplus_matrices])
        compute_bader_ofer = theano.function(inputs=[seats, votes, surplus_matrices], outputs=comp_bo)
        
        kosher_votes = trace.sum(axis=2,keepdims=True)
        
        passed_votes = trace / kosher_votes
        passed_votes[passed_votes < threshold] = 0
        
        initial_moded = (passed_votes.sum(axis=2, keepdims=True) / 120)
        initial_seats = (passed_votes // initial_moded).astype('int64')
    
        ndays = trace.shape[1]
        nparties = trace.shape[2]
        
        if surpluses is None:
            surpluses = self.create_surplus_matrices()

        return compute_bader_ofer(initial_seats, passed_votes,
            surpluses * np.ones([ndays, nparties, nparties], dtype='int64'))
        
    def compute_trace_bader_ofer_segmented(self, trace, surpluses = None, threshold = None, segment_size=1000):
        traces = np.split(trace, [i * segment_size for i in range(trace.shape[0] // segment_size) if i is not 0])
        return np.concatenate([ self.compute_trace_bader_ofer(t, surpluses, threshold) for t in traces ], axis=0)

    def get_least_square_sum_seats(self, bader_ofer, day=0, segment_size=1000):
        """
        Determine the sample whose average distance in seats to the other samples
        is most minimal, distance computed as the square root of sum of squares
        of the seats of the parties.
        """
        bo_plot = bader_ofer.transpose(1,2,0)[day]
        bo_plots_a = np.split(bo_plot[:,:,None], [ i * segment_size for i in range(bo_plot.shape[1] // segment_size) if i != 0 ], axis=1)
        bo_plots_b = np.split(bo_plot[:,None,:], [ i * segment_size for i in range(bo_plot.shape[1] // segment_size) if i != 0 ], axis=2)
        bo_sqrsums = [ np.concatenate([ np.sqrt(np.sum((bpa - bpb) ** 2, axis=0)).sum(axis=0) for bpb in bo_plots_b ]) for bpa in bo_plots_a ]
        return bader_ofer[np.stack(bo_sqrsums).sum(axis=0).argmin()][0]
    
    def compute_interval(self, values, alpha=0.95):    
        avg = values.mean()
        scale = values.std()
        
        return ss.norm.interval(alpha, avg, scale)

    def compute_mandates_interval(self, mandates, alpha=0.95):    
        threshold = float(self.forecast_model.config['threshold_percent']) / 100

        def convert_interval(i):
          if i < int(threshold * 120):
            return 0
          else:
            return np.round(i)
        
        return tuple(convert_interval(i) for i in self.compute_interval(mandates, alpha))
      
    def plot_mandates(self, bader_ofer, max_bo=None, day=0, hebrew=True, subtitle='', segment_size=1000):
        """
        Plot the resulting mandates of the parties and their distributions.
        This is the bar graph most often seen in poll results.
        """
        
        from bidi import algorithm as bidialg
        
        fe=self.forecast_model
        parties = fe.parties
    
        bo_plot = bader_ofer.transpose(1,2,0)[day]
    
        if max_bo is None:
            max_bo = self.get_least_square_sum_seats(bader_ofer, day, segment_size=segment_size)

        num_passed_parties = len(np.where(max_bo > 0)[0])
        passed_parties = max_bo.argsort()[::-1]
        fig, plots = plt.subplots(2, num_passed_parties, figsize=(2 * num_passed_parties, 10), gridspec_kw={'height_ratios':[5,1]} )
        fig.set_facecolor('white')
        xlim_dists = []
        ylim_height = []
        max_bo_height = max_bo.max()
        for i in range(num_passed_parties) :
          party = passed_parties[i]
          name = bidialg.get_display(parties[fe.party_ids[party]]['hname']) if hebrew else parties[fe.party_ids[party]]['name']
          plots[0][i].set_title(name, va='bottom', y=-0.08, fontsize='large')
          mandates_count = np.unique(bo_plot[party], return_counts=True)
          mandates_bar = plots[0][i].bar([0], [max_bo[party]])[0] #, tick_label=[name])
          plots[0][i].set_xlim(-0.65,0.65)
          plots[0][i].text(mandates_bar.get_x() + mandates_bar.get_width()/2.0, mandates_bar.get_height(), '%d' % max_bo[party], ha='center', va='bottom', fontsize='x-large')
          plots[0][i].set_ylim(top=max_bo_height)
          bars = plots[1][i].bar(mandates_count[0], 100 * mandates_count[1] / len(bo_plot[party]))
          xticks = []
          xtick_labels = []
          max_start = 0
          if 0 in mandates_count[0]:
            #xticks += [0]
            #xtick_labels += [ '' ]
            max_start = 1
            zero_rect = bars[0]
            zero_rect.set_color('red')
            plots[1][i].text(zero_rect.get_x() + zero_rect.get_width()/2.0, zero_rect.get_height(), ' %d%%' % (100 * mandates_count[1][0] / len(bo_plot[party])), ha='center', va='bottom')
          if len(mandates_count[1]) > max_start:
              interval = self.compute_mandates_interval(bo_plot[party])
              max_index = max_start + np.argmax(mandates_count[1][max_start:])
              max_rect = bars[max_index]
              #plots[1][i].text(max_rect.get_x() + max_rect.get_width()/2.0, max_rect.get_height(), ' %d%%' % (100 * mandates_count[1][max_index] / len(bo_plot[party])), ha='center', va='bottom')
              xticks += [mandates_count[0][max_index]]
              xtick_labels += [ '\n%d - %d' % interval ]
          plots[1][i].set_xticks(xticks)
          plots[1][i].set_xticklabels(xtick_labels)
          xlim = plots[1][i].get_xlim()
          xlim_dists += [ xlim[1] - xlim[0] + 1 ]
          ylim_height += [ plots[1][i].get_ylim()[1] ]
          plots[0][i].grid(False)
          plots[0][i].tick_params(axis='both', which='both',left=False,bottom=False,labelbottom=False,labelleft=False)
          plots[0][i].set_facecolor('white')
          for s in plots[0][i].spines.values():
            s.set_visible(False)
          plots[1][i].grid(False)
          plots[1][i].tick_params(axis='y', which='both',left=False,labelleft=False)
          plots[1][i].set_facecolor('white')
          for s in plots[1][i].spines.values():
            s.set_visible(False)
        xlim_side = max(xlim_dists) / 2
        for i in range(num_passed_parties) :
          xlim = plots[1][i].get_xlim()
          xlim_center = (xlim[0] + xlim[1]) / 2
          plots[1][i].set_xlim(xlim_center - xlim_side, xlim_center + xlim_side)
          plots[1][i].set_ylim(top=max(ylim_height))
        bo_mean = bo_plot.mean(axis=1)
        failed_parties = [ i for i in bo_mean.argsort()[::-1] if max_bo[i] == 0 ]
        num_failed_parties = len(failed_parties)
        offset = num_passed_parties // 3
        failed_plots = []
        for failed_index in range(num_failed_parties):
            failed_plot = fig.add_subplot(num_failed_parties*2, num_passed_parties,
                num_passed_parties * (failed_index + 1) - offset, ymargin = 1)
            party = failed_parties[failed_index]
            mandates_count = np.unique(bo_plot[party], return_counts=True)
            if len(mandates_count[0]) > 1:
                max_start = 0
                xticks = []
                xtick_labels = []
                bars = failed_plot.bar(mandates_count[0], 100 * mandates_count[1] / len(bo_plot[party]))
                if 0 in mandates_count[0]:
                    #xticks += [0]
                    max_start = 1
                    zero_rect = bars[0]
                    zero_rect.set_color('red')
                    failed_plot.text(zero_rect.get_x() + zero_rect.get_width()/2.0, zero_rect.get_height(), ' %d%%' % (100 * mandates_count[1][0] / len(bo_plot[party])), ha='center', va='bottom')
                if len(mandates_count[1]) > max_start:
                    interval = self.compute_mandates_interval(bo_plot[party])
                    max_index = max_start + np.argmax(mandates_count[1][max_start:])
                    max_rect = bars[max_index]
                    #failed_plot.text(max_rect.get_x() + max_rect.get_width()/2.0, max_rect.get_height(), ' %d%%' % (100 * mandates_count[1][max_index] / len(bo_plot[party])), ha='center', va='bottom')
                    xticks += [mandates_count[0][max_index]]
                    xtick_labels += [ '\n%d - %d' % interval ]
                failed_plot.set_xticks(xticks)
                failed_plot.set_xticklabels(xtick_labels)
            else:
                failed_plot.text(0.8, 0.5, str(mandates_count[0][0]), ha='center', va='bottom')
            name = bidialg.get_display(parties[fe.party_ids[party]]['hname']) if hebrew else parties[fe.party_ids[party]]['name']
            failed_plot.set_ylabel(name, va='center', ha='right', rotation=0, fontsize='medium')
            failed_plot.yaxis.set_label_position("right")
            failed_plot.spines["right"].set_position(("axes", 1.25))
            failed_plot.grid(False)
            failed_plot.tick_params(axis='both', which='both',left=False,bottom=False,labelbottom=False,labelleft=False)
            failed_plot.set_facecolor('white')
            for s in failed_plot.spines.values():
              s.set_visible(False)
            failed_plots += [ failed_plot ]
        if num_failed_parties > 0:
            max_failed_xlim = max([fp.get_xlim()[1] for fp in failed_plots])
            for fp in failed_plots:
                fp.set_xlim(right=max_failed_xlim)
                fp.set_ylim(0, 150)
                
        title = 'חלוקת המנדטים' if hebrew else 'Mandates Allocation'
        if subtitle != '':
            title += ' - ' + subtitle
        if hebrew:
            title = bidialg.get_display(title)
            
        fig.text(.5, 1.05, title, ha='center', fontsize='xx-large')
        if fe.house_effects_model is not None:
            fig.text(.5, 1., self.house_effects_model_title(hebrew), ha='center', fontsize='small')
        fig.figimage(self.create_logo(), fig.bbox.xmax / 2 + 50, fig.bbox.ymax - 100, zorder=1000)

    def plot_coalitions(self, bader_ofer, coalitions=None, day=0, min_mandates_for_coalition=61, stable_mandates_for_coalition=65, hebrew=True, coalitions_shape=None, subtitle=''):
        """
        Plot the resulting mandates of the coalitions and their distributions.
        """
    
        from bidi import algorithm as bidialg
    
        fe=self.forecast_model
    
        if coalitions is None:
          coalitions = fe.config['coalitions']
    
        num_coalitions = len(coalitions)
        coalitions_matrix = np.zeros([num_coalitions, fe.num_parties], dtype='bool')
        for i, (coalition, config) in enumerate(coalitions.items()):
           for party in config['parties']:
              party_index = fe.party_ids.index(party)
              coalitions_matrix[i][party_index] = 1
    
        bo_plot = bader_ofer.transpose(1,2,0)[day]
        coalitions_mandates = coalitions_matrix[:,:,None] * bo_plot[None,:,:].astype('int64')
        for i, (coalition, config) in enumerate(coalitions.items()):
          if 'subsets' in config:
             for party, subset_config in config['subsets'].items():
                party_index = fe.party_ids.index(party)
                cum_members = np.asarray([sum(j <= i for j in subset_config['members']) for i in range(1 + max(subset_config['members']))])
                coalitions_mandates[i][party_index] = cum_members[coalitions_mandates[i][party_index]]
        coalitions_bo = coalitions_mandates.sum(axis=1)
    
        if coalitions_shape is None:
            coalitions_shape = (1, num_coalitions)
        fig, plots = plt.subplots(coalitions_shape[0], coalitions_shape[1], figsize=(5 * coalitions_shape[1], 5 * coalitions_shape[0]))
        fig.set_facecolor('white')
        xlim_dists = []
        ylim_height = []
        
        colors = [
          '#ff0000', # 57 = red
          '#ff3d00', # 58
          '#ff7900', # 59
          '#ffb600', # 60
          '#fff200', # 61 = yellow
          '#c7db00', # 62
          '#8fc400', # 63
          '#56ad00', # 64
          '#1e9600', # 65 = green
        ]
        
        coalition_names = sorted([ bidialg.get_display(config['hname']) if hebrew else config['name'] for config in coalitions.values() ], key=lambda p: p[::-1] if hebrew else p, reverse=hebrew)
        for i, (coalition, config) in enumerate(coalitions.items()) :
          name = bidialg.get_display(config['hname']) if hebrew else config['name']
          coalition_index = coalition_names.index(name)
          if coalitions_shape[0] > 1:
            plot = plots[coalition_index // coalitions_shape[1], coalition_index % coalitions_shape[1]]
          else:
            plot = plots[coalition_index]
          title = plot.set_title(name, va='bottom', y=-0.2, fontsize='large')
          def get_party_name(party):
            namevar = 'hname' if hebrew else 'name'
            name = fe.parties[party][namevar]
            if 'subsets' in config and party in config['subsets']:
                name += ' (%s)' % config['subsets'][party][namevar]
            return bidialg.get_display(name) if hebrew else name

          party_names = [ get_party_name(party) for party in config['parties'] ]
          plot.text(0.5, -0.2, '\n'.join(sorted(party_names, key=lambda p: p[::-1] if hebrew else p)),
               ha='center', va='top', fontsize='small', transform=plot.transAxes)
          mandates_count = np.unique(coalitions_bo[i], return_counts=True)
          bars = plot.bar(mandates_count[0], 100 * mandates_count[1] / len(coalitions_bo[i]))
          for mandates, bar in zip(mandates_count[0], bars):
            cindex = int(min(8, max(0, mandates - 57)))
            bar.set_color(colors[cindex])
              
          xticks = []
          max_start = 0
    
          if 0 in mandates_count[0]:
            xticks += [0]
            max_start = 1
    
          mean_mandates = coalitions_bo[i].mean()
          if len(mandates_count[1]) > max_start:
              mean_index = np.where(mandates_count[0]==int(np.round(mean_mandates)))
              mean_index = mean_index[0][0]
              mean_rect = bars[mean_index]
              xticks += [mandates_count[0][mean_index]]
          num_minimum = len(np.where(coalitions_bo[i] >= min_mandates_for_coalition)[0])
          num_stable =len(np.where(coalitions_bo[i] >= stable_mandates_for_coalition)[0])
          perc_text = plot.text(mean_rect.get_x() + mean_rect.get_width()/2.0, mean_rect.get_y() + mean_rect.get_height()/3.0, '%.1f%%' % (100 * num_minimum / coalitions_bo[i].shape[0]), ha='center', va='top', fontsize='xx-large', fontweight='bold')
          perc_text.set_path_effects([pe.withStroke(linewidth=4, foreground='w', alpha=0.7)])
          perc_title = plot.text(mean_rect.get_x() + mean_rect.get_width()/2.0, mean_rect.get_y() + mean_rect.get_height()/3.0,
              bidialg.get_display('%d ומעלה:' % min_mandates_for_coalition) if hebrew else '%d and Above:' % min_mandates_for_coalition,
              ha='center', va='bottom', fontsize='medium', fontweight='bold')
          perc_title.set_path_effects([pe.withStroke(linewidth=4, foreground='w', alpha=0.7)])
          perc_text = plot.text(mean_rect.get_x() + mean_rect.get_width()/2.0, mean_rect.get_y() + 2*mean_rect.get_height()/3.0, '%.1f%%' % (100 * num_stable / coalitions_bo[i].shape[0]), ha='center', va='top', fontsize='xx-large', fontweight='bold')
          perc_text.set_path_effects([pe.withStroke(linewidth=4, foreground='w', alpha=0.7)])
          perc_title = plot.text(mean_rect.get_x() + mean_rect.get_width()/2.0, mean_rect.get_y() + 2*mean_rect.get_height()/3.0,
              bidialg.get_display('%d ומעלה:' % stable_mandates_for_coalition) if hebrew else '%d and Above:' % stable_mandates_for_coalition,
              ha='center', va='bottom', fontsize='medium', fontweight='bold')
          perc_title.set_path_effects([pe.withStroke(linewidth=4, foreground='w', alpha=0.7)])
          plot.set_xticks(xticks)
          xlim = plot.get_xlim()
          xlim_dists += [ xlim[1] - xlim[0] + 1 ]
          ylim_height += [ plot.get_ylim()[1] ]
          plot.grid(False)
          plot.tick_params(axis='y', which='both',left=False,labelleft=False)
          plot.set_facecolor('white')
          for s in plot.spines.values():
            s.set_visible(False)
        xlim_side = max(xlim_dists) / 2
        for i in range(num_coalitions) :
          xlim = plot.get_xlim()
          xlim_center = (xlim[0] + xlim[1]) / 2
          plot.set_xlim(xlim_center - xlim_side, xlim_center + xlim_side)
          plot.set_ylim(top=max(ylim_height))
    
        title = 'קואליציות' if hebrew else 'Coalitions'
        if subtitle != '':
            title += ' - ' + subtitle
        if hebrew:
            title = bidialg.get_display(title)
    
        fig.text(.5, 1.05, title, ha='center', fontsize='xx-large')
        if fe.house_effects_model is not None:
            fig.text(.5, 1., self.house_effects_model_title(hebrew), ha='center', fontsize='small')
        fig.figimage(self.create_logo(), fig.bbox.xmax / 2 + 100, fig.bbox.ymax - 0, zorder=1000)

    def plot_pollster_house_effects(self, samples, hebrew = True):
        """
        Plot the house effects of each pollster per party.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import matplotlib.ticker as ticker
        from bidi import algorithm as bidialg
        
        house_effects = samples.transpose(2,1,0)
        fe = self.forecast_model
        
        actual_pollsters = [i for i in fe.dynamics.pollster_mapping.items() if i[1] is not None]
        pollster_ids = [fe.pollster_ids[pollster] for _,pollster in sorted(actual_pollsters, key=lambda i: i[1])]

        plots = []
        for i, party in enumerate(fe.party_ids):
          def pollster_label(pi, pollster_id):
              perc = '%.2f %%' % (100 * house_effects[i][pi].mean())
              if hebrew and len(fe.config['pollsters'][pollster_id]['hname']) > 0:
                  label = perc + ' :' + bidialg.get_display(fe.config['pollsters'][pollster_id]['hname'])
              else:
                  label = fe.config['pollsters'][pollster_id]['name'] + ': ' + perc
              return label
            
          cpalette = sns.color_palette("cubehelix", len(pollster_ids))
          patches = [
              mpatches.Patch(color=cpalette[pi], label=pollster_label(pi, pollster))
              for pi, pollster in enumerate(pollster_ids)]
    
          fig, ax = plt.subplots(figsize=(10, 2))
          fig.set_facecolor('white')
          legend = fig.legend(handles=patches, loc='best', ncol=2)
          if hebrew:
            for col in legend._legend_box._children[-1]._children:
                for c in col._children: 
                    c._children.reverse() 
                col.align="right" 
          ax.set_title(bidialg.get_display(fe.parties[party]['hname']) if hebrew 
                       else fe.parties[party]['name'])
          for pi, pollster_house_effects in enumerate(house_effects[i]):
            sns.kdeplot(100 * pollster_house_effects, shade=True, ax=ax, color=cpalette[pi])
          ax.xaxis.set_major_formatter(ticker.PercentFormatter(decimals=1))
          ax.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=1))
          plots += [ax]
          fig.text(.5, 1.05, bidialg.get_display('הטיית הסוקרים') if hebrew else 'House Effects', 
                   ha='center', fontsize='xx-large')
          fig.text(.5, .05, 'Generated using pyHoshen © 2019 - 2021', ha='center')

    def plot_party_support_evolution_graphs(self, samples, mbo = None, burn=None, hebrew = True):
        """
        Plot the evolving support of each party over time in both percentage and seats.
        """
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        import matplotlib.dates as mdates
        from bidi import algorithm as bidialg
    
        def get_dimensions(n):
            divisor = int(np.ceil(np.sqrt(n)))
            if n % divisor == 0:
                return n // divisor, divisor
            else:
                return 1 + n // divisor, divisor
     
        if burn is None:
            burn = -min(len(samples), 1000)
            
        if mbo is None:
            mbo = self.compute_trace_bader_ofer(samples)
    
        samples = samples[burn:]
        mbo = mbo[burn:]
        
        fe = self.forecast_model
                
        mbo_by_party = np.transpose(mbo, [2, 1, 0])
        mandates = 0
        mbo_by_day_party = np.transpose(mbo, [1, 2, 0])
        party_avg = mbo_by_day_party[0].mean(axis=1).argsort()[::-1]
        date_list = [fe.forecast_day - datetime.timedelta(days=x) for x in range(0, fe.num_days)]
        dimensions = get_dimensions(fe.num_parties)
        fig, plots = plt.subplots(dimensions[1], dimensions[0], 
                                  figsize=(5.5 * dimensions[0], 3.5 * dimensions[1]))
        fig.set_facecolor('white')
        means = samples.mean(axis=0)
        stds = samples.std(axis=0)
    
        for index, party in enumerate(party_avg):
            party_config = fe.config['parties'][fe.party_ids[party]]
            
            if 'created' in party_config:
                days_to_show = (fe.forecast_day - strpdate(party_config['created'])).days
            else:
                days_to_show = fe.num_days

            vindex = index // dimensions[0]
            hindex = index % dimensions[0]
            if hebrew:
                hindex = -hindex - 1 # work right to left in hebrew
            subplots = plots[vindex]
            
            subplots[hindex].xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
    
            title = bidialg.get_display(party_config['hname']) if hebrew else party_config['name']
            subplots[hindex].set_title(title)
    
            subplot = subplots[hindex].twinx()
        
            num_day_ticks = fe.num_days
            date_ticks = [fe.forecast_day - datetime.timedelta(days=x) 
                for x in range(0, num_day_ticks, 7)]
            subplots[hindex].set_xticks(date_ticks)
            subplots[hindex].set_xticklabels(subplots[hindex].get_xticklabels(), rotation=45)
            subplots[hindex].set_xlim(date_list[-1], date_list[0])
            subplot.set_xticks(date_ticks)
            subplot.set_xticklabels(subplot.get_xticklabels(), rotation=45)
            subplot.set_xlim(date_list[-1], date_list[0])
    
            party_means = means[:, party]
            party_std = stds[:, party]
            
            subplots[hindex].fill_between(date_list[:days_to_show],
                    100*party_means[:days_to_show] - 100*1.95996*party_std[:days_to_show],
                    100*party_means[:days_to_show] + 100*1.95996*party_std[:days_to_show],
                    color='#90ee90')
            subplots[hindex].plot(date_list[:days_to_show],
                    100*party_means[:days_to_show], color='#32cd32')
            if subplots[hindex].get_ylim()[0] < 0:
                subplots[hindex].set_ylim(bottom=0)
            subplots[hindex].yaxis.set_major_formatter(ticker.PercentFormatter(decimals=1))
            subplots[hindex].tick_params(axis='y', colors='#32cd32')
            subplots[hindex].yaxis.label.set_color('#32cd32')
                
            mand_means = np.mean(mbo_by_party[party], axis=1)
            mand_std = np.std(mbo_by_party[party], axis=1)
            subplot.fill_between(date_list[:days_to_show],
                    mand_means[:days_to_show] - 1.95996*mand_std[:days_to_show], 
                    mand_means[:days_to_show] + 1.95996*mand_std[:days_to_show], 
                    alpha=0.5, color='#6495ed')
            subplot.plot(date_list[:days_to_show], mand_means[:days_to_show], color='#4169e1')
            if subplot.get_ylim()[0] < 0:
                subplot.set_ylim(bottom=0)
            if subplot.get_ylim()[1] < 4:
                subplot.set_ylim(top=4)
            if int(subplot.get_ylim()[1]) == int(subplot.get_ylim()[0]):
                subplot.set_ylim(top=int(subplot.get_ylim()[0]) + 1)
            subplot.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, min_n_ticks=2, prune=None))
            subplot.tick_params(axis='y', colors='#4169e1')
            subplot.yaxis.label.set_color('#4169e1')
                                          
            subplots[hindex].yaxis.tick_right()
            subplots[hindex].yaxis.set_label_position("right")
            subplots[hindex].spines["right"].set_position(("axes", 1.08))
            
            for s in subplots[hindex].spines.values():
              s.set_visible(False)
            for s in subplot.spines.values():
              s.set_visible(False)

            if hindex == dimensions[0] - 1:
                subplot.set_ylabel("Seats")
                subplots[hindex].set_ylabel("% Support")
                subplots[hindex].spines["right"].set_position(("axes", 1.13))
            elif hindex == -1:
                subplot.set_ylabel(bidialg.get_display("מנדטים"))
                subplots[hindex].set_ylabel(bidialg.get_display("אחוזי תמיכה"))
                subplots[hindex].spines["right"].set_position(("axes", 1.13))
     
            subplots[hindex].grid(False)           
            
            sup = [sample[party][0] for sample in samples]
            mandates += int(120.0*sum(sup)/len(samples))
            
            subplots[hindex].xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
            
            subplots[hindex].set_xlim(date_ticks[-1], date_ticks[0])
            subplot.set_xlim(date_ticks[-1], date_ticks[0])
    
        for empty_subplot in range(len(party_avg), np.product(dimensions)):
            hindex = empty_subplot % dimensions[0]
            if hebrew:
                hindex = -hindex - 1 # work right to left in hebrew
            plots[-1][hindex].axis('off')
        plt.subplots_adjust(wspace=0.5,hspace=0.5)
    
        if hebrew:
            title = bidialg.get_display('התמיכה במפלגות לאורך זמן')
        else:
            title = 'Party Support over Time'
            
        fig.text(.5, 1.05, title, ha='center', fontsize='xx-large')
        if fe.house_effects_model is not None:
            fig.text(.5, 1., self.house_effects_model_title(hebrew), ha='center', fontsize='small')

        fig.figimage(self.create_logo(), fig.bbox.xmax / 2 + 100, fig.bbox.ymax - 100, zorder=1000)

    def plot_correlation_matrix(self, correlation_matrix, hebrew=False):
        """
        Plot the given correlation matrix.
        """
        from bidi import algorithm as bidialg
        
        labels = [bidialg.get_display(v['hname']) if hebrew else v['name']
            for v in self.forecast_model.parties.values()]
    
        utils.plot_correlation_matrix(correlation_matrix, labels, alignRight=hebrew)

    def plot_election_correlation_matrices(self, correlation_matrices, hebrew=False):
        """
        Plot the distribution of correlation matrices.
        """
        from bidi import algorithm as bidialg
    
        labels = [bidialg.get_display(v['hname'])  if hebrew else v['name'] 
            for v in self.forecast_model.parties.values()]
        fig = utils.plot_correlation_matrices(correlation_matrices, labels, alignRight=hebrew)
        fig.set_facecolor('white')
        fig.text(.5, 1.05, bidialg.get_display('מטריצת המתאמים') if hebrew else 'Correlation Matrix', 
                 ha='center', fontsize='xx-large')
        fig.text(.5, .05, 'Generated using pyHoshen © 2019 - 2021', ha='center')

