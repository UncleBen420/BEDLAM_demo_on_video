import os
import sys
import argparse
from loguru import logger
from glob import glob
from train.core.testerx import Tester

os.environ['PYOPENGL_PLATFORM'] = 'egl'
sys.path.append('')

import cv2
import os

def split_video_to_images(video_path, output_folder, frame_skip=1):
    """
    Splits a video into JPEG images and saves them to a specified folder.

    Parameters:
        video_path (str): Path to the input video file.
        output_folder (str): Path to the folder where extracted images will be saved.
        frame_skip (int): Number of frames to skip between each saved image. Defaults to 1 (save every frame).

    Returns:
        None
    """
    # Check if the output folder exists; if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Capture the video
    video_capture = cv2.VideoCapture(video_path)

    if not video_capture.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    frame_index = 0
    saved_count = 0

    while True:
        ret, frame = video_capture.read()
        if not ret:
            # Break the loop if no frame is retrieved (end of video)
            break

        if frame_index % frame_skip == 0:
            # Construct the file name for the frame
            filename = os.path.join(output_folder, f"frame_{frame_index:04d}.jpg")

            # Save the frame as a JPEG file
            cv2.imwrite(filename, frame)
            saved_count += 1

        frame_index += 1

    video_capture.release()
    print(f"Done! {saved_count} frames were saved to {output_folder}")

# Example Usage
# split_video_to_images('example.mp4', 'output_frames', frame_skip=10)


def main(args):

    input_image_folder = args.image_folder
    output_path = args.output_folder
    os.makedirs(output_path, exist_ok=True)

    logger.add(
        os.path.join(output_path, 'demo.log'),
        level='INFO',
        colorize=False,
    )
    logger.info(f'Demo options: \n {args}')

    tester = Tester(args)
    
    video_files = [v for v in os.listdir(input_image_folder) if v.endswith(".mp4")]

    for v in video_files:
        print(v)
        os.makedirs(os.path.join(input_image_folder, v.split('.')[0] + '_res'), exist_ok=True)

        split_video_to_images(os.path.join(input_image_folder, v),
                              os.path.join(input_image_folder, v.split('.')[0]))
        detections = tester.run_detector([os.path.join(input_image_folder, v.split('.')[0])])
        tester.run_on_image_folder([os.path.join(input_image_folder, v.split('.')[0])],
                                   detections, os.path.join(input_image_folder, v.split('.')[0] + '_res'), args.display, args.save_result, from_one_video=True)

    del tester.model

    logger.info('================= END =================')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', type=str, default='configs/demo_bedlam_cliff_x.yaml',
                        help='config file that defines model hyperparams')

    parser.add_argument('--ckpt', type=str, default='data/ckpt/bedlam_cliff_x.ckpt',
                        help='checkpoint path')

    parser.add_argument('--image_folder', type=str, default='demo_videos',
                        help='input image folder')

    parser.add_argument('--output_folder', type=str, default='demo_images/results',
                        help='output folder to write results')

    parser.add_argument('--tracker_batch_size', type=int, default=1,
                        help='batch size of object detector used for bbox tracking')
                        
    parser.add_argument('--display', action='store_true',
                        help='visualize the 3d body projection on image')

    parser.add_argument('--detector', type=str, default='yolo', choices=['yolo', 'maskrcnn'],
                        help='object detector to be used for bbox tracking')

    parser.add_argument('--yolo_img_size', type=int, default=416,
                        help='input image size for yolo detector')
    parser.add_argument('--eval_dataset', type=str, default=None)
    parser.add_argument('--dataframe_path', type=str, default=None)
    parser.add_argument('--data_split', type=str, default='test')
    parser.add_argument('--save_result', action='store_true', help='Save verts, joints, joints2d in pkl file to evaluate')

    args = parser.parse_args()
    main(args)
