# build dataset according to given 'dataset_file'
def build_dataset(args):
    if args.dataset_file == 'SHHA':
        from crowd_datasets.SHHA.loading_data import loading_data
        return loading_data

    return None

def build_dataset_partial(args):
    if args.dataset_file == 'SHHA_partial':
        from crowd_datasets.SHHA.loading_data import loading_data_partial
        return loading_data_partial

def build_dataset_unsup(args):
    if args.dataset_file == 'SHHA_partial':
        from crowd_datasets.SHHA.loading_data import loading_data_unsup
        return loading_data_unsup