import os
import glob
import argparse
import random
import fnmatch
import json
import sys

def find_recursive(root_dir, ext='.mp3'):
    files = []
    for root, dirnames, filenames in os.walk(root_dir):
        for filename in fnmatch.filter(filenames, '*' + ext):
            files.append(os.path.join(root, filename))
    return files


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_audio', default='./data/audio',
                        help="root for extracted audio files")
    parser.add_argument('--root_frame', default='./data/frames',
                        help="root for extracted video frames")
    parser.add_argument('--fps', default=8, type=int,
                        help="fps of video frames")
    parser.add_argument('--path_output', default='./data',
                        help="path to output index files")
    parser.add_argument('--trainset_ratio', default=0.8, type=float,
                        help="80% for training, 20% for validation")
    parser.add_argument('--json_files', default='MUSIC_solo_videos.json', type=str,
                        help="path to input json file")
    args = parser.parse_args()
    
    json_files = [args.json_files]
    
    vid2cat = {}
    cat2vids = {}
    all_videos = set()
    
    for curr_file in json_files:
      data = json.load(open(curr_file))
      videos = data['videos']
      for cat in videos:
        cat_videos = videos[cat]
        for vid in cat_videos:
          vid2cat[vid] = cat
        if cat not in cat2vids:
          cat2vids[cat] = set()
        cat2vids[cat] = cat2vids[cat].union(set(cat_videos))
        
    valid_videos = set(vid2cat.keys())

    # find all audio/frames pairs
    infos = []
    audio_files = find_recursive(args.root_audio, ext='.wav')
    for audio_path in audio_files:
        vid_name = audio_path.split('/')[-1]
        vid_name = vid_name.split('.wav')[0]
        
        if vid_name not in valid_videos:
            continue
    
        frame_path = audio_path.replace(args.root_audio, args.root_frame) \
                               .replace('.wav', '.mp4')
        frame_files = glob.glob(frame_path + '/*.jpg')
        if len(frame_files) > args.fps * 20:
            infos.append(','.join([audio_path, frame_path, str(len(frame_files))]))
    print('{} audio/frames pairs found.'.format(len(infos)))

    # split train/val
    n_train = int(len(infos) * 0.8)
    random.shuffle(infos)
    trainset = infos[0:n_train]
    valset = infos[n_train:]
    for name, subset in zip(['music_only_train', 'music_only_val'], [trainset, valset]):
        filename = '{}.csv'.format(os.path.join(args.path_output, name))
        with open(filename, 'w') as f:
            for item in subset:
                f.write(item + '\n')
        print('{} items saved to {}.'.format(len(subset), filename))

    print('Done!')
