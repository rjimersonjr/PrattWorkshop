import sys, argparse, json, os, librosa
import soundfile as sf

def main(argv):
    parser = argparse.ArgumentParser(description='Grab lines for images')
    parser.add_argument('-csvFile','--csvFile', help='file', required=True)
    parser.add_argument('-manifestFile','--manifestFile', help='file', required=True)
    args = parser.parse_args()
    argsdict = vars(args)
    csvFile = argsdict['csvFile']
    manifestFile = argsdict['manifestFile']
    csvFileOpen = open(csvFile, 'r')
    Lines = csvFileOpen.readlines()
    with open(manifestFile, 'w', encoding='utf8') as fout:
        for line in Lines:
            arrLine = line.split(",")
            audio_path = arrLine[0].strip()
            transcript = arrLine[1].strip()
            duration = librosa.core.get_duration(filename=audio_path)
            metadata = {
                "audio_filepath": audio_path,
                "duration": duration,
                "text": transcript
            }
            print(metadata)
            json.dump(metadata, fout, ensure_ascii=False)
            fout.write('\n')

if __name__ == "__main__":
    main(sys.argv[1:])
