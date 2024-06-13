class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def save_results(self, file_path, df):
    with open(file_path, 'a') as file:
        file.write('\n' + self.args.model_name + ' ' + self.args.validation +'\n')
        df.to_csv(file, header=True, index=False)