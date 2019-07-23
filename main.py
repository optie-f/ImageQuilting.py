from Modules.ImageQuilting import ImageQuilting
import cv2
import time
import datetime
import os


def main():
    o_size = 512
    b_size = 32
    imdir = './tex/'
    imgs = os.listdir(imdir)
    for (i, imfile) in enumerate(imgs):
        if imfile[0] == '.':
            continue

        print(imfile, ':')
        tex = cv2.imread(imdir + imfile)
        imageQuilting = ImageQuilting(tex)

        ct = time.time()
        imageQuilting(outputSize=o_size, blockSize=b_size)
        print('input:{0}, out:{1}, patch: {2}'.format(tex.shape[:2], o_size, b_size))
        print('runtime: ', datetime.timedelta(seconds=time.time() - ct))
        cv2.imwrite('./result/{0}_size{1}_patch{2}.jpg'.format(imfile.split('.')[0], o_size, b_size), imageQuilting.output)
        cv2.imwrite('./result/{0}_size{1}_patch{2}_line.jpg'.format(imfile.split('.')[0], o_size, b_size), imageQuilting.patchLine)


if __name__ == '__main__':
    main()
