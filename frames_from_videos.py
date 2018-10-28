import cv2
import glob
import os
import argparse
import logging

logging.basicConfig(filename="logs/frames_from_videos.log", filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s', datefmt='%H:%M:%S',
                    level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler())


def extractImages(pathIn, pathOut, fileName, frameDistance):
    """Extract frames from videos at given legnth of frameDistance"""
    videoName = pathIn.split("\\")[-1].split(".")[0].encode("utf-8")
    logging.info("Extracting frames from the video: " + str(videoName))
    vidcap = cv2.VideoCapture(pathIn)
    success, image = vidcap.read()
    count = 0
    success = True
    while success:
        success, image = vidcap.read()
        if count % frameDistance == 0:
            cv2.imwrite(pathOut + "\\" + str(fileName) + "_frame%d.jpg" % count, image)  # save frame as JPEG file
        count += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get frames from Videos; supports mp4 videos')
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument('-f', '--folder', default=None, type=str, help='folder path to the video')
    parser.add_argument('-d', '--frameDistance', default=250, type=int,
                        help='Number of frames to skip between two saves')
    requiredNamed.add_argument('-s', '--save', default=None, type=str, help='folder path to save the frames')

    args = parser.parse_args()
    video_folder = None
    save_folder = None
    frameDistance = 250
    if not args.folder or not args.save:
        parser.print_help()
        raise parser.error('Input/output folder not given!')
    if args.folder:
        video_folder = args.folder
    if args.frameDistance:
        frameDistance = args.frameDistance
    if args.save:
        save_folder = args.save

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    file_count = 0
    for file in glob.glob(os.path.join(video_folder, "*.mp4")):
        extractImages(file, save_folder, file_count, frameDistance)
        file_count += 1
