import os
import torch
import numpy as np
import json


def set_deterministic(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    # torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)   
    np.random.seed(seed)      

def print_config_description(conf_path):
    with open(conf_path, 'r') as f:
        config_dict = json.load(f)
    description = config_dict.get("description", None)
    if description is not None:
        print(description)
    else:
        print("Configuration file does not contain a description")
        print("We highly recommend you to add a description to the configuration file for the debugging")
        

def load_audio_and_label_file_paths(args):

    # audio_path = './data/' + args.corpus + '/Audios'
    # video_path = './data/' + args.corpus + '/Videos'

    audio_path = './data/IEMOCAP/wav'

    video_path = './data/IEMOCAP/npy'

    if args.corpus == 'MSP-IMPROV':
        if args.data_mode == 'primary':
            if args.num_classes == 'four':
                label_path = './data/' + args.corpus + '/Partitioned_data_Primary_Emotion/labels_consensus_EmoP_4class_' + args.label_rule + '/labels_consensus_' + args.partition_number + '.csv'
        elif args.data_mode == 'secondary':
            if args.num_classes == 'four':
                label_path = './data/' + args.corpus + '/Partitioned_data_Secondary_Emotion/labels_consensus_EmoS_4class_' + args.label_rule + '/labels_consensus_' + args.partition_number + '.csv'
            else:
                label_path = './data/' + args.corpus + '/Partitioned_data_Secondary_Emotion/labels_consensus_EmoS_10class_' + args.label_rule + '/labels_consensus_' + args.partition_number + '.csv'


    elif args.corpus == 'CREMA-D':
        if args.num_classes == 'four':
            label_path = './data/' + args.corpus + '/labels_consensus_4class_' + args.label_rule + '/labels_consensus_' + args.partition_number + '.csv'
        else:
            # label_path = './data/' + args.corpus + '/labels_consensus_6class_' + args.label_rule + '/label_consensus_' + args.partition_number + '.csv'
            label_path = './data/labels_consensus_6class_M/label_consensus_1.csv'

    elif args.corpus == 'IEMOCAP':
        if args.num_classes == 'four':
            label_path = './data/IEMOCAP/label2/split_{}.csv'.format(args.fold)
        else:
            print("label error")


    return audio_path, video_path, label_path
