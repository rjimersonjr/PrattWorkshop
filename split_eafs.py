#!/usr/bin/python3
#
# Copyright Ben Foley ben@cbmm.io 30 Jan 2018
#
# Split an audio file by the start and end times of annotations on a particular .eaf tier
# Don't worry about 'Parsing unknown version of ELAN spec... ' warnings,
# pympi is looking for v 2.7 or 2.8 of elan schema

# You can open and save WAV files with pure python.
# For opening and saving non-wav files – like mp3 – you'll need ffmpeg or libav.
# On OSX, install ffmpeg with `brew install ffmpeg`


# default usage: python3 split_eafs.py

import argparse
import glob
import json
import os
import sys
import re
from pydub import AudioSegment
from pympi.Elan import Eaf
from slugify import slugify


parser = argparse.ArgumentParser(description="This script will slice audio and output text based on ELAN annotations.")
parser.add_argument('-i', '--input_dir', help='Directory of audio and eaf files', type=str, default='./input')
parser.add_argument('-o', '--tier_order', help='Get the annotations from this tier index, eg top tier would be 1', type=int, default='1')
parser.add_argument('-t', '--tier', help='Target language tier name', type=str, default='Seneca')
parser.add_argument('-m', '--silence_marker', help='Skip any annotations on the target language tier with this value', type=str, default='*PUB')
parser.add_argument('-s', '--silence_tier', help='Silence audio when annotations are found on this ref tier', type=str, default='Silence')
parser.add_argument('-a', '--output_audio_dir', help='Directory to save the audio files', type=str, default='./output')
parser.add_argument('-l', '--output_label_dir', help='Directory to save text files', type=str, default='./output')
parser.add_argument('-j', '--output_json', help='File name to output json', type=str, default='./output/annotations.json')
parser.add_argument('-v', '--verbose', help='Verbose output', action='store_true')
parser.add_argument('-n', '--name_with_annotation', help='Name the file with annotation value', action='store_true')
parser.add_argument('-p', '--prefix', help='Prefix the file name with this', type=str, default='')
parser.add_argument('-f', '--output_audio_format', help='Audio format to save media clips (requires ffmpeg)', type=str, default='wav')
parser.add_argument('-q', '--eaf_file', help='Specific file to use', type=str)

args = parser.parse_args()
try:
    input_dir = args.input_dir
    tier_order = args.tier_order
    tier = args.tier
    silence_marker = args.silence_marker
    silence_tier = args.silence_tier
    output_audio_dir = args.output_audio_dir
    output_label_dir = args.output_label_dir
    output_json = args.output_json
    verbose = args.verbose
    name_with_annotation = args.name_with_annotation
    prefix = args.prefix
    output_audio_format = args.output_audio_format.lower()
    eaf_file = args.eaf_file
except Exception:
    parser.print_help()
    sys.exit(0)

# Limit to wav or mp3 outputs
valid_formats = ['mp3', 'wav']
if output_audio_format not in valid_formats:
    raise Exception('Please use wav or mp3 output formats only')

if not os.path.exists(output_audio_dir):
    os.makedirs(output_audio_dir)
if not os.path.exists(output_label_dir):
    os.makedirs(output_label_dir)

if verbose:
    print("tier_order", tier_order)
    print("tier", tier)


def split_audio_by_start_end(input_audio, start, end, fname):
    output = input_audio[start:end]
    output.export(os.path.join(output_audio_dir, fname + "." + output_audio_format), format=output_audio_format)


def write_text(annotation, fname):
    f = open(os.path.join(output_label_dir, fname + ".txt"), 'w')
    f.write(annotation)
    f.close()


def write_json(annotations_data):
    with open(output_json, 'w') as outfile:
        json.dump(annotations_data, outfile, indent=4, separators=(',', ': '), sort_keys=False)


def read_eaf(ie):

    #print("Are we in read_eaf?")
    if verbose:
        print("input file is", ie)

    input_eaf = Eaf(ie)

    # Check if the tiers we have been given exist
    tier_names = list(input_eaf.get_tier_names())
    if verbose:
        print("tier_names", tier_names, file=sys.stderr)

    # Are we working by tier name or order?
    if tier != "default" :
        if verbose:
            print("using tier by name:", tier, file=sys.stderr)
    else:

        # Sanity check that the tier num is not greater than the num of tiers
        if tier_order > len(tier_names):
            print("Error: tier number is greater than the number of tiers",
                  file=sys.stderr)
            return False
        if verbose:
            print("using tier by number:", tier_names[tier_order-1], file=sys.stderr)

    if tier not in tier_names:
        print('Error: missing tier ' + tier, file=sys.stderr)
        return False
    if silence_tier not in tier_names:
        if verbose:
            print('silence tier not found: ' + silence_tier, file=sys.stderr)

    # get the input audio file
    inDir, name = os.path.split(ie)
    basename, ext = os.path.splitext(name)
    # we can write out mp3 or whatever, still require wav input
    ia = os.path.join(inDir, basename + ".wav")
    input_audio = AudioSegment.from_wav(ia)

    # We can pass in an arg for a ref tier that has silence labels
    check_silence_ref_tier = False
    if silence_tier in tier_names:
        silence_tier_info = input_eaf.get_parameters_for_tier(silence_tier)
        if silence_tier_info.get("PARENT_REF") == tier:
            check_silence_ref_tier = True

    # Get annotation values, start and end times, and speaker id
    annotations = sorted(input_eaf.get_annotation_data_for_tier(tier))
    params = input_eaf.get_parameters_for_tier(tier)
    if 'PARTICIPANT' in params:
        speaker_id = params['PARTICIPANT']

    annotations_data = []
    i = 0
    for ann in annotations:
        skip = False
        ref_annotation = []
        start = ann[0]
        end = ann[1]
        # output new values, not the original clip start end times
        clip_start = 0
        clip_end = ann[1] - ann[0]
        annotation = ann[2]
        #print("Is this the annotation: " + str(annotation))
        annotation = annotation.lower()

        # Check for annotations labelled with a particular symbol on the main tier
        if annotation == silence_marker:
            skip = True

        # Check for existence of an annotation in ref tier to silence
        # Annotation value doesn't matter
        if check_silence_ref_tier:
            ref_annotation = input_eaf.get_ref_annotation_at_time(silence_tier, start)
            if len(ref_annotation) is True:
                skip = True

        if skip is True:
            print('skipping annotation: ' + annotation, start, end)
        else:
            #print('processing annotation: ' + annotation, start, end)
            # build the output audio/text filename
            fname = basename + "_" + str(i)
            if name_with_annotation:
                fname = slugify(annotation)

            if prefix != '':
                fname = prefix + '_' + fname
            #print("What is audio directory: " + str(output_audio_dir))
            current_working_directory = os.getcwd()
            print(current_working_directory + "/output/"  + fname + ".wav," + annotation)
            #print(fname + "," + annotation)
            obj = {
                #'audioFileName': os.path.join(".", fname + ".wav"),
                'audioFileName': os.path.join(output_audio_dir, fname + ".wav"),
                'transcript': annotation,
                'startMs': clip_start,
                'stopMs': clip_end
            }
            if 'PARTICIPANT' in params:
                obj["speakerId"] = speaker_id
            annotations_data.append(obj)
            split_audio_by_start_end(input_audio, start, end, fname)
            write_text(annotation, fname)
            i += 1
    # output the json data for the next step in kaldi pipeline
    write_json(annotations_data)


    if verbose:
        print(annotations_data)


def findFilesByExt(setOfAllFiles, exts):
    res = []
    for f in setOfAllFiles:
        name, ext = os.path.splitext(f)
        if ("*" + ext.lower()) in exts:
            res.append(f)
    return res


g_exts = ["*.eaf"]
allFilesInDir = set(glob.glob(os.path.join(input_dir, "**"), recursive=True))
#print("What are all the files in the dir: " + str(allFilesInDir))
input_eafs = findFilesByExt(allFilesInDir, set(g_exts))

read_eaf(eaf_file)

#for ie in input_eafs:
#    read_eaf(ie)
