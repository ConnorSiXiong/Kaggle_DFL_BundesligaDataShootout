import cv2
import os


def extract_images(video_path, out_dir):
    video_name = os.path.basename(video_path).split('.')[0]
    video_reader = cv2.VideoCapture(video_path)

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    w = 100
    h = 100
    fps = 20.0

    out = cv2.VideoWriter('output.mp4', fourcc, fps, (w, h))  # 形状得一样

    print(video_path)

    frame_count = 0

    running = True

    while running:

        successed, frame = video_reader.read()

        resize_frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)  # 形状得一样
        out.write(resize_frame)

        if not successed:
            break
        frame_count += 1
        if frame_count % 2 == 0:
            continue
        outfile = f'{out_dir}/{video_name}_{frame_count:06}.jpg'
        img = cv2.resize(frame, dsize=(100, 100), interpolation=cv2.INTER_AREA)
        cv2.imwrite(outfile, img)

        if frame_count >= 20:
            running = False
    out.release()


if __name__ == '__main__':
    out_dir = '..'
    video_path = '../data/cut_1.mp4'
    extract_images(video_path, '..')
