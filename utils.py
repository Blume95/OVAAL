import logging


def create_logger(log_file=None):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.propagate = False

    return logger


def build_dataloader():
    1

def build_model():
    1


def define_dataset(mode, used_class, args):
    if args.dataset == "cs":
        dataset = CityscapesDataset(mode, used_class, args)
    elif args.dataset == "cv":
        dataset = CamVid(mode, used_class, args)
    elif args.dataset == 'voc':
        dataset = BerkeleyDeepDrive(mode, used_class, args)
    else:
        raise TypeError("Not exist dataset")
    return dataset