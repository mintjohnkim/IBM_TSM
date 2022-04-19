

DATASET_CONFIG = {
    'st2stv2': {
        'num_classes': 174,
        'train_list_name': 'train.txt',
        'val_list_name': 'val.txt',
        'test_list_name': 'test.txt',
        'filename_seperator': " ",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 3
    },
    'mini_st2stv2': {
        'num_classes': 87,
        'train_list_name': 'mini_train.txt',
        'val_list_name': 'mini_val.txt',
        'test_list_name': 'mini_test.txt',
        'filename_seperator': " ",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 3
    },
    'kinetics400': {
        'num_classes': 400,
        'train_list_name': 'train.txt',
        'val_list_name': 'val.txt',
        'test_list_name': 'test.txt',
        'filename_seperator': ";",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 30
    },
    'mini_kinetics400': {
        'num_classes': 200,
        'train_list_name': 'mini_train.txt',
        'val_list_name': 'mini_val.txt',
        'test_list_name': 'mini_test.txt',
        'filename_seperator': ";",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 30
    },
    'moments': {
        'num_classes': 339,
        'train_list_name': 'train.txt',
        'val_list_name': 'val.txt',
        'filename_seperator': " ",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 0
    },
    'mini_moments': {
        'num_classes': 200,
        'train_list_name': 'mini_train.txt',
        'val_list_name': 'mini_val.txt',
        'filename_seperator': " ",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 0
    },
    'epickitchens55verb': {
        'num_classes': 125,
        'train_list_name': 'epic_train_verb.txt',
        'val_list_name': 'epic_val_verb.txt',
        'filename_seperator': ";",
        'image_tmpl': 'frame_{:010d}.jpg',
        'filter_video': 0
    },
    'epickitchens55action': {
        'num_classes': 2513,
        'train_list_name': 'epic_train.txt',
        'val_list_name': 'epic_val.txt',
        'filename_seperator': ";",
        'image_tmpl': 'frame_{:010d}.jpg',
        'filter_video': 0
    },
    'epickitchens55actionmanyshot': {
        'num_classes': 819,
        'train_list_name': 'epic_train_manyshot.txt',
        'val_list_name': 'epic_val_manyshot.txt',
        'filename_seperator': ";",
        'image_tmpl': 'frame_{:010d}.jpg',
        'filter_video': 0
    },
    'ucf101': {
        'num_classes': 101,
        'train_list_name': 'train.txt',
        'val_list_name': 'val.txt',
        'filename_seperator': ";",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 30
    },
    'charadesego': {
        'num_classes': 157,
        'train_list_name': 'charades_train_combined.txt',
        'val_list_name': 'charades_val.txt',
        'filename_seperator': ";",
        'image_tmpl': '{:06d}.jpg',
        'filter_video': 30
    },
    'mini_sim': {
        'num_classes': 43,
        'train_list_name': 'mini_sim_train.txt',
        'val_list_name': 'mini_sim_val.txt',
        'filename_seperator': ";",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 0
    },
    'mini_sim_v2': {
        'num_classes': 38,
        'train_list_name': 'mini_sim_train.txt',
        'val_list_name': 'mini_sim_val.txt',
        'filename_seperator': ";",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 0
    },
    'moments_sub': {
        'num_classes': 38,
        'train_list_name': 'moment_sub_train.txt',
        'val_list_name': 'moment_sub_val.txt',
        'filename_seperator': ";",
        'image_tmpl': '{:06d}.jpg',
        'filter_video': 0
    },
    'moments_sub_38c_1000e': {
        'num_classes': 38,
        'train_list_name': 'moment_sub_38c_1000e_train.txt',
        'val_list_name': 'moment_sub_38c_val.txt',
        'filename_seperator': ";",
        'image_tmpl': '{:06d}.jpg',
        'filter_video': 0
    },
    'moments_sub_50c_1000e': {
        'num_classes': 50,
        'train_list_name': 'moment_sub_50c_1000e_train.txt',
        'val_list_name': 'moment_sub_50c_val.txt',
        'filename_seperator': ";",
        'image_tmpl': '{:06d}.jpg',
        'filter_video': 0
    },
    'moments_sub_100c_1000e': {
        'num_classes': 100,
        'train_list_name': 'moment_sub_100c_1000e_train.txt',
        'val_list_name': 'moment_sub_100c_val.txt',
        'filename_seperator': ";",
        'image_tmpl': '{:06d}.jpg',
        'filter_video': 0
    },
    'moments_sub_150c_1000e': {
        'num_classes': 150,
        'train_list_name': 'moment_sub_150c_1000e_train.txt',
        'val_list_name': 'moment_sub_150c_val.txt',
        'filename_seperator': ";",
        'image_tmpl': '{:06d}.jpg',
        'filter_video': 0
    },
    'moments_sub_200c_1000e': {
        'num_classes': 200,
        'train_list_name': 'moment_sub_200c_1000e_train.txt',
        'val_list_name': 'moment_sub_200c_val.txt',
        'filename_seperator': ";",
        'image_tmpl': '{:06d}.jpg',
        'filter_video': 0
    },
    'moments_sub_200c_250e': {
        'num_classes': 200,
        'train_list_name': 'moment_sub_200c_250e_train.txt',
        'val_list_name': 'moment_sub_200c_val.txt',
        'filename_seperator': ";",
        'image_tmpl': '{:06d}.jpg',
        'filter_video': 0
    },
    'moments_sub_200c_500e': {
        'num_classes': 200,
        'train_list_name': 'moment_sub_200c_500e_train.txt',
        'val_list_name': 'moment_sub_200c_val.txt',
        'filename_seperator': ";",
        'image_tmpl': '{:06d}.jpg',
        'filter_video': 0
    },
    'moments_sub_200c_750e': {
        'num_classes': 200,
        'train_list_name': 'moment_sub_200c_750e_train.txt',
        'val_list_name': 'moment_sub_200c_val.txt',
        'filename_seperator': ";",
        'image_tmpl': '{:06d}.jpg',
        'filter_video': 0
    },
    'moments_sub_38c_2000e': {
        'num_classes': 38,
        'train_list_name': 'moment_sub_38c_2000e_train.txt',
        'val_list_name': 'moment_sub_38c_val.txt',
        'filename_seperator': ";",
        'image_tmpl': '{:06d}.jpg',
        'filter_video': 0
    },
    'moments_sub_38c_3000e': {
        'num_classes': 38,
        'train_list_name': 'moment_sub_38c_3000e_train.txt',
        'val_list_name': 'moment_sub_38c_val.txt',
        'filename_seperator': ";",
        'image_tmpl': '{:06d}.jpg',
        'filter_video': 0
    },
    'moments_sub_100c_1000e': {
        'num_classes': 100,
        'train_list_name': 'moment_sub_100c_1000e_train.txt',
        'val_list_name': 'moment_sub_100c_val.txt',
        'filename_seperator': ";",
        'image_tmpl': '{:06d}.jpg',
        'filter_video': 0
    },
    'moments_sub_366c': {
        'num_classes': 366,
        'train_list_name': 'moment_sub_366c_-1e_train.txt',
        'val_list_name': 'moment_sub_366c_val.txt',
        'filename_seperator': ";",
        'image_tmpl': '{:06d}.jpg',
        'filter_video': 0
    },
    'kinetics100': {
        'num_classes': 100,
        'train_list_name': 'kinetics100train.txt',
        'val_list_name': 'kinetics100val.txt',
        'test_list_name': 'test.txt',
        'filename_seperator': ";",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 30
    },
    'kinetics200': {
        'num_classes': 200,
        'train_list_name': 'kinetics200train.txt',
        'val_list_name': 'kinetics200val.txt',
        'test_list_name': 'test.txt',
        'filename_seperator': ";",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 30
    },
    'kinetics300': {
        'num_classes': 300,
        'train_list_name': 'kinetics300train.txt',
        'val_list_name': 'kinetics300val.txt',
        'test_list_name': 'test.txt',
        'filename_seperator': ";",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 30
    },
    # 'kinetics400_250': {
    #     'num_classes': 400,
    #     'train_list_name': 'kinetics400_250e_train.txt',
    #     'val_list_name': 'kinetics400val.txt',
    #     'test_list_name': 'test.txt',
    #     'filename_seperator': ";",
    #     'image_tmpl': '{:05d}.jpg',
    #     'filter_video': 30
    # },
    # 'kinetics400_500': {
    #     'num_classes': 400,
    #     'train_list_name': 'kinetics400_500e_train.txt',
    #     'val_list_name': 'kinetics400val.txt',
    #     'test_list_name': 'test.txt',
    #     'filename_seperator': ";",
    #     'image_tmpl': '{:05d}.jpg',
    #     'filter_video': 30
    # },
    # 'kinetics400_750': {
    #     'num_classes': 400,
    #     'train_list_name': 'kinetics400_750e_train.txt',
    #     'val_list_name': 'kinetics400val.txt',
    #     'test_list_name': 'test.txt',
    #     'filename_seperator': ";",
    #     'image_tmpl': '{:05d}.jpg',
    #     'filter_video': 30
    # },
    'kinetics30_1000': {
        'num_classes': 30,
        'train_list_name': 'kinetics30_1000e_train.txt',
        'val_list_name': 'kinetics30val.txt',
        'test_list_name': 'test.txt',
        'filename_seperator': ";",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 30
    },
    'kinetics50_1000': {
        'num_classes': 50,
        'train_list_name': 'kinetics50_1000e_train.txt',
        'val_list_name': 'kinetics50val.txt',
        'test_list_name': 'test.txt',
        'filename_seperator': ";",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 30
    },
    'kinetics60_1000': {
        'num_classes': 60,
        'train_list_name': 'kinetics60_1000e_train.txt',
        'val_list_name': 'kinetics60val.txt',
        'test_list_name': 'test.txt',
        'filename_seperator': ";",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 30
    },
    'kinetics90_1000': {
        'num_classes': 90,
        'train_list_name': 'kinetics90_1000e_train.txt',
        'val_list_name': 'kinetics90val.txt',
        'test_list_name': 'test.txt',
        'filename_seperator': ";",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 30
    },
    'kinetics100_1000': {
        'num_classes': 100,
        'train_list_name': 'kinetics100_1000e_train.txt',
        'val_list_name': 'kinetics100val.txt',
        'test_list_name': 'test.txt',
        'filename_seperator': ";",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 30
    },
    'kinetics120_1000': {
        'num_classes': 120,
        'train_list_name': 'kinetics120_1000e_train.txt',
        'val_list_name': 'kinetics120val.txt',
        'test_list_name': 'test.txt',
        'filename_seperator': ";",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 30
    },
    'kinetics150_250': {
        'num_classes': 150,
        'train_list_name': 'kinetics150_250e_train.txt',
        'val_list_name': 'kinetics150val.txt',
        'test_list_name': 'test.txt',
        'filename_seperator': ";",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 30
    },
    'kinetics150_500': {
        'num_classes': 150,
        'train_list_name': 'kinetics150_500e_train.txt',
        'val_list_name': 'kinetics150val.txt',
        'test_list_name': 'test.txt',
        'filename_seperator': ";",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 30
    },
    'kinetics150_750': {
        'num_classes': 150,
        'train_list_name': 'kinetics150_750e_train.txt',
        'val_list_name': 'kinetics150val.txt',
        'test_list_name': 'test.txt',
        'filename_seperator': ";",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 30
    },
    'kinetics150_1000': {
        'num_classes': 150,
        'train_list_name': 'kinetics150_1000e_train.txt',
        'val_list_name': 'kinetics150val.txt',
        'test_list_name': 'test.txt',
        'filename_seperator': ";",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 30
    },
    'kinetics200_1000': {
        'num_classes': 200,
        'train_list_name': 'kinetics200_1000e_train.txt',
        'val_list_name': 'kinetics200val.txt',
        'test_list_name': 'test.txt',
        'filename_seperator': ";",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 30
    },
    'kinetics200_250': {
        'num_classes': 200,
        'train_list_name': 'kinetics200_250e_train.txt',
        'val_list_name': 'kinetics200val.txt',
        'test_list_name': 'test.txt',
        'filename_seperator': ";",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 30
    },
    'kinetics200_500': {
        'num_classes': 200,
        'train_list_name': 'kinetics200_500e_train.txt',
        'val_list_name': 'kinetics200val.txt',
        'test_list_name': 'test.txt',
        'filename_seperator': ";",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 30
    },
    'kinetics200_750': {
        'num_classes': 200,
        'train_list_name': 'kinetics200_750e_train.txt',
        'val_list_name': 'kinetics200val.txt',
        'test_list_name': 'test.txt',
        'filename_seperator': ";",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 30
    },
    # 'sim50_1000': {
    #     'num_classes': 50,
    #     'train_list_name': 'sim50_1000e_train.txt',
    #     'val_list_name': 'sim50_val.txt',
    #     'test_list_name': 'test.txt',
    #     'filename_seperator': ";",
    #     'image_tmpl': '{:05d}.jpg',
    #     'filter_video': 0
    # },
    'sim30_1000': {
        'num_classes': 30,
        'train_list_name': 'sim30_1000e_train.txt',
        'val_list_name': 'sim30_val.txt',
        'test_list_name': 'test.txt',
        'filename_seperator': ";",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 0
    },
    'sim60_1000': {
        'num_classes': 60,
        'train_list_name': 'sim60_1000e_train.txt',
        'val_list_name': 'sim60_val.txt',
        'test_list_name': 'test.txt',
        'filename_seperator': ";",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 0
    },
    'sim90_1000': {
        'num_classes': 90,
        'train_list_name': 'sim90_1000e_train.txt',
        'val_list_name': 'sim90_val.txt',
        'test_list_name': 'test.txt',
        'filename_seperator': ";",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 0
    },
    'sim120_1000': {
        'num_classes': 120,
        'train_list_name': 'sim120_1000e_train.txt',
        'val_list_name': 'sim120_val.txt',
        'test_list_name': 'test.txt',
        'filename_seperator': ";",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 0
    },
    'sim150_1000': {
        'num_classes': 150,
        'train_list_name': 'sim150_1000e_train.txt',
        'val_list_name': 'sim150_val.txt',
        'test_list_name': 'test.txt',
        'filename_seperator': ";",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 0
    },
    'sim150_250': {
        'num_classes': 150,
        'train_list_name': 'sim150_250e_train.txt',
        'val_list_name': 'sim150_val.txt',
        'test_list_name': 'test.txt',
        'filename_seperator': ";",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 0
    },
    'sim150_500': {
        'num_classes': 150,
        'train_list_name': 'sim150_500e_train.txt',
        'val_list_name': 'sim150_val.txt',
        'test_list_name': 'test.txt',
        'filename_seperator': ";",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 0
    },
    'sim150_750': {
        'num_classes': 150,
        'train_list_name': 'sim150_750e_train.txt',
        'val_list_name': 'sim150_val.txt',
        'test_list_name': 'test.txt',
        'filename_seperator': ";",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 0
    },
    'mix_sim40kinetics110_1000': {
        'num_classes': 150,
        'train_list_name': 'mix_sim40kinetics110_1000e_train.txt',
        'val_list_name': 'mix_sim40kinetics110_val.txt',
        'test_list_name': 'test.txt',
        'filename_seperator': ";",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 0
    },
    'mix_sim75kinetics75_1000': {
        'num_classes': 150,
        'train_list_name': 'mix_sim75kinetics75_1000e_train.txt',
        'val_list_name': 'mix_sim75kinetics75_val.txt',
        'test_list_name': 'test.txt',
        'filename_seperator': ";",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 0
    },
    'mix_sim110kinetics40_1000': {
        'num_classes': 150,
        'train_list_name': 'mix_sim110kinetics40_1000e_train.txt',
        'val_list_name': 'mix_sim110kinetics40_val.txt',
        'test_list_name': 'test.txt',
        'filename_seperator': ";",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 0
    },
    'diving48': {
        'num_classes': 48,
        'train_list_name': 'Diving48_V2_train.txt',
        'val_list_name': 'Diving48_V2_test.txt',
        'test_list_name': 'test.txt',
        'filename_seperator': ",",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 0
    },
    'hmdb51': {
        'num_classes': 51,
        'train_list_name': 'hmdb_train.txt',
        'val_list_name': 'hmdb_val.txt',
        'test_list_name': 'test.txt',
        'filename_seperator': ";;",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 0
    },
    'somethingsomethingv2': {
        'num_classes': 174,
        'train_list_name': 'mini_train.txt',
        'val_list_name': 'mini_val.txt',
        'test_list_name': 'test.txt',
        'filename_seperator': " ",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 0
    },
    'ikea': {
        'num_classes': 12,
        'train_list_name': 'ikea_train.txt',
        'val_list_name': 'ikea_val.txt',
        'test_list_name': 'test.txt',
        'filename_seperator': ";",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 0
    },
    'uav': {
        'num_classes': 155,
        'train_list_name': 'uav_train.txt',
        'val_list_name': 'uav_val.txt',
        'test_list_name': 'test.txt',
        'filename_seperator': ";",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 0
    },
}


def get_dataset_config(dataset):
    ret = DATASET_CONFIG[dataset]
    num_classes = ret['num_classes']
    train_list_name = ret['train_list_name']
    val_list_name = ret['val_list_name']
    test_list_name = ret.get('test_list_name', None)
    filename_seperator = ret['filename_seperator']
    image_tmpl = ret['image_tmpl']
    filter_video = ret.get('filter_video', 0)
    label_file = ret.get('label_file', None)

    return num_classes, train_list_name, val_list_name, test_list_name, filename_seperator, \
           image_tmpl, filter_video, label_file
