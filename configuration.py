import numpy as np
import datetime
import pandas as pd
import json
import io
import os
import os.path

def get_version():
    return 2

class Configuration:
    def __init__(self, config):
        self.dataframes = {}
        self.config, self.path = self.read_config(config)
        
    def __getitem__(self, key):
        return self.config[key]
        
    def read_config(self, config):
        max_filesize = 1024 * 1024
        assert type(config) in [dict, str], 'expected either str or dict'
        config_path = os.getcwd()

        if type(config) is dict:
            return config, config_path
        
        if '://' in config:
            import urllib
            config_data = urllib.request.urlopen(config).read(max_filesize)
        elif config.endswith('json'):
            config_path = os.path.dirname(os.path.abspath(config))
            with open(config, 'r', encoding='utf8') as f:
                config_data = f.read(max_filesize)
        else:
            config_data = config

        if config.endswith('.hjson'):
            import hjson
            return hjson.loads(config_data), config_path
        elif config.endswith('.json'):
            import json
            return json.loads(config_data), config_path
        else:
            try:
                import hjson
                return hjson.loads(config_data), config_path
            except ImportError:
                import json
                return json.loads(config_data), config_path
                
    def read_config_datasets(self, dataframes, config):
        for dataset, data in config.items():
            if 'type' in data:
                if data['type'] == 'google-sheets':
                    import gspread
                    from oauth2client.client import GoogleCredentials
                    gc = gspread.authorize(GoogleCredentials.get_application_default())                    
                    gs = gc.open_by_key(data['key'])
                    df = pd.DataFrame(gs.worksheet(data['worksheet']).get_all_records())
                    if 'parse_dates' in data:
                        for col in data['parse_dates']:
                            df[col] = pd.to_datetime(df[col],
                              format=data['date_format'])
                else:
                    raise Exception('unknown type: %s' % data['type'])
            else:
                filename = os.path.normpath(os.path.join(self.path, data['filename']))
                extension = os.path.splitext(filename)[1]
                if extension == '.csv':
                    parse_dates = data['parse_dates'] if 'parse_dates' in data else False
                    if 'date_format' in data:
                        date_parser = lambda x: datetime.datetime.strptime(x, data['date_format']).date()
                    else:
                        date_parser = None
                    encoding = data['encoding'] if 'encoding' in data else None
                    df = pd.read_csv(filename, encoding=encoding,
                                     parse_dates=parse_dates, date_parser=date_parser)
                elif extension == '.xls' or extension == '.xlsx':
                    header = data['header'] if 'header' in data else 0
                    nrows = data['nrows'] if 'nrows' in data else None
                    index_col = data['index_col'] if 'index_col' in data else None
                    usecols = data['usecols'] if 'usecols' in data else None
                    df = pd.read_excel(filename, header=header, nrows=nrows,
                                index_col=index_col,
                                skip_rows=range(min(header)))
                    if usecols is not None:
                        df = df[[df.columns[i] for i in usecols]]
                else:
                    raise Exception('unknown extension: %s' % extension)
        
            df.columns = df.columns.to_series().apply(lambda x: 
                ' '.join(str(c) for c in x if 'Unnamed' not in str(c)) if type(x) is tuple else x)
            
            df.columns = df.columns.to_series().apply(lambda x: x.strip())
            if 'columns' in data:
                df = df[list(data['columns'].values())]
        
                df.columns = data['columns'].keys()
    
            if 'groupby' in data:
                df = df.groupby(data['index']).sum().reset_index()
            if 'groupby' in data or data['index'] in df.columns:
                df.set_index(data['index'])
            else:
                df.index.name = data['index']
                df[data['index']] = df.index
    
            if 'dropna' in data:
                if 'dropna_subset' in data:
                    df = df.dropna(how=data['dropna'],subset=data['dropna_subset'])
                else:
                    df = df.dropna(how=data['dropna'])
            if 'output' in data:
                df = df.eval('\n'.join('%s = %s' % (k, v) for k,v in data['output'].items()))
                output = list(data['output'].keys())
            else:
                output = df.columns
            df = df[output]
            if df.index.name != data['index']:
                df.set_index(data['index'])
            dataframes[dataset] = df
    
    def read_data(self, configs):
        for category, category_filename in configs.items():
            self.dataframes[category] = {}
            filename = os.path.normpath(os.path.join(self.path, category_filename))
            with io.open(filename, 'r', encoding='utf-8-sig') as f:
                config = json.load(f)
                self.read_config_datasets(self.dataframes[category], config)

    def read_polls(self, cycle_config, polls):
        if 'polls' not in self.dataframes:
            self.dataframes['polls'] = {}
        for category, poll_config in polls.items():
            if poll_config['type'] == 'csv':
                self.read_config_datasets(self.dataframes['polls'], 
                    { category: 
                        {'encoding': 'utf-8', 'filename': poll_config['filename'],
                         'index': 'id', 'parse_dates': ['start_date'], 
                         'date_format': '%Y-%m-%d'}
                    })
            else:
                poll_config['index'] = 'id'
                poll_config['parse_dates'] = ['start_date']
                poll_config['date_format'] = '%Y-%m-%d'
                self.read_config_datasets(self.dataframes['polls'], 
                    { category:  poll_config })
                
            df = self.dataframes['polls'][category]
            parties = [ p for p in df.columns if p.startswith('p_') ]
            if 'method' in poll_config:
                if poll_config['method'] == 'deduce':
                    total_seats = poll_config['total_seats']
                    others_col = poll_config['others_col']
                    others_full = np.float64(poll_config['others_full'])
                    others_min = np.float64(poll_config['others_min'])
                    threshold = poll_config['threshold']
                    default_num_polled = int(poll_config['default_num_polled'])
                    impute_missing = 'impute_missing' in poll_config and poll_config['impute_missing'].lower() in [ 'yes', 'true', '1' ]
                    party_config = cycle_config['parties']
                    party_inits = { p: datetime.datetime.strptime(party_config[p]['created'], '%d/%m/%Y') for p in party_config if 'created' in party_config[p] }
                    party_dests = { p: datetime.datetime.strptime(party_config[p]['dissolved'], '%d/%m/%Y') for p in party_config if 'dissolved' in party_config[p] }
                    party_unions = { p: cycle_config['parties'][p]['union_of'] 
                        for p in parties 
                        if p in cycle_config['parties']
                        and 'union_of' in cycle_config['parties'][p] }
                    new_columns = [ c for c in df.columns ]
                    #print (new_columns)
                    new_parties = parties.copy()
                    for composite, components in party_unions.items():
                        if composite in new_parties:
                            pass
                        elif others_col in new_parties:
                            new_columns.insert(new_columns.index(others_col), composite)
                            new_parties.insert(new_parties.index(others_col), composite)
                        else:
                            new_columns += [ composite ]
                            new_parties += [ composite ]
                        for c in components:
                            if c != composite:
                                new_columns.remove(c)
                                new_parties.remove(c)
                    #print (new_columns)
                    new_rows = []
                    for i, row in df.iterrows():
                      if len(row['pollster']) == 0:
                          break
                      mands = {}
                      percs = {}
                      for p in parties:
                        if type(row[p]) is int:
                          mands[p] = int(row[p])
                        elif type(row[p]) is float:
                          mands[p] = float(row[p])
                        elif row[p].endswith('%'):
                          percs[p] = np.float64(row[p][:-1])/100
                        elif len(row[p]) > 0:
                          raise ValueError("invalid row element at row %d: %s" % (i, row[p]))
                      assert round(sum(mands.values()),3) >= total_seats, "not enough mandates in row %d, sum = %.3f: %s" % (i, sum(mands.values()), str(mands))
                      if others_col in mands or others_col in percs:
                        others = 0
                      elif len(percs) > 0 or sum(mands.values()) > total_seats:
                        others = others_min
                      else:
                        others = others_full
                      if sum(mands.values()) > total_seats:
                        too_low = [p for p, m in mands.items() if m/total_seats < threshold]
                        percs.update({p: np.float64(mands[p])/total_seats for p in too_low })
                        for p in too_low:
                          del mands[p]
                      normalize_to = 1.0 - sum(percs.values()) - others
                    #  print (i, sum(mands.values()), normalize_to, sum(percs.values()))
                      percs.update({ p: m * normalize_to / total_seats for p, m in mands.items() })
                      num_days = max(1, row['num_days'])
                      for p in parties:
                          if p != others_col and p not in percs:
                              if p in party_inits and row['start_date'] <= party_inits[p]:
                                  percs[p] = 0
                              if p in party_dests and row['start_date'] + datetime.timedelta(days=num_days) >= party_dests[p]:
                                  percs[p] = 0
                      for p in percs:
                          assert type(p) is str, "type of %s is not str, but %s" % (p, type(p))
                      #print (i, percs)
                      assert abs(sum(percs.values()) + others - 1.0) < 0.001, "didn't add up to 1.0! %f" % sum(percs.values())
                      num_missing = sum(1 for p in parties if p not in percs and p != others_col)
                      if not impute_missing and num_missing > 0:
                          for p in parties:
                              if p not in percs:
                                  percs[p] = others_full / num_missing
                          percs[others_col] = others_min
                          others = 0
                          total_percs = sum(percs.values())
                          for p in percs:
                              percs[p] /= total_percs
                      for composite, components in party_unions.items():
                          percs[composite] = sum(percs[c] for c in components)
                      if others_col in percs:
                          percs[others_col] = others_min
                      new_row = {}
                      for c in new_columns:
                          if c in new_parties:
                              new_row[c] = np.float64(percs[c]) if c in percs else np.nan
                          elif c == 'start_date':
                              new_row[c] = row['start_date']
                          else:
                              new_row[c] = row[c]
                      if type(new_row['num_polled']) is not int:
                          if 'computed_num' in new_row:
                              new_row['num_polled'] = new_row['computed_num']
                      if type(new_row['num_polled']) is not int:
                          new_row['num_polled'] = default_num_polled
                      new_row['num_days'] = num_days
                      #print ([ new_row[c] for c in new_columns ])
                      new_rows += [[ new_row[c] for c in new_columns ]]
                    newdf = pd.DataFrame(new_rows, columns=new_columns)
                    #newdf[new_parties] = newdf[new_parties].astype(float)
                    #newdf['start_date'] = pd.to_datetime(newdf['start_date'])
                    #newdf = newdf.append(new_rows)
                    self.dataframes['polls'][category] = newdf
                else:
                    raise Exception("unknown method %s" % poll_config['method'])
            else:
                self.dataframes['polls'][category] = df.eval('\n'.join(['{0}={0}/120'.format(p) for p in parties]))
            