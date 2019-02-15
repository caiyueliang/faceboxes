# coding=utf-8
import cv2
import os
import time
import shutil
import common as common

# ['EVENT_FLAG_ALTKEY', 'EVENT_FLAG_CTRLKEY', 'EVENT_FLAG_LBUTTON', 'EVENT_FLAG_MBUTTON', 'EVENT_FLAG_RBUTTON',
# 'EVENT_FLAG_SHIFTKEY', 'EVENT_LBUTTONDBLCLK', 'EVENT_LBUTTONDOWN', 'EVENT_LBUTTONUP', 'EVENT_MBUTTONDBLCLK',
# 'EVENT_MBUTTONDOWN', 'EVENT_MBUTTONUP', 'EVENT_MOUSEHWHEEL', 'EVENT_MOUSEMOVE', 'EVENT_MOUSEWHEEL',
# 'EVENT_RBUTTONDBLCLK', 'EVENT_RBUTTONDOWN', 'EVENT_RBUTTONUP']

# events = [i for i in dir(cv2) if 'EVENT' in i]
# img = np.zeros((512, 512, 3), np.uint8)


# mouse callback function
# def mouse_click_events(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDBLCLK:
#         cv2.circle(img, (x, y), 100, (255, 0, 0), -1)


class FilterImage:
    def __init__(self, image_dir_1, image_dir_2, output_dir, index_file):
        self.image_dir_1 = image_dir_1
        self.img_files_1 = common.get_files(image_dir_1)

        self.image_dir_2 = image_dir_2
        self.img_files_2 = common.get_files(image_dir_2)

        print(len(self.img_files_1), len(self.img_files_2))

        self.output_dir = output_dir
        self.index_file = index_file
        return

    # def mouse_click_events(self, event, x, y, flags, param):
    #     if event == cv2.EVENT_LBUTTONDOWN:
    #         if len(self.car_points) < 4:
    #             cv2.circle(self.img, (x, y), 3, (0, 0, 255), -1)
    #             print('click: [%d, %d]' % (x, y))
    #             self.car_points.append((x, y))
    #         else:
    #             print('self.car_points is too long, %s' % str(self.car_points))

    def filter_start(self, restart=False):
        times = 1
        start_i = 0

        if restart is False:
            try:
                start_i = int(common.read_data(self.index_file, 'r'))
                print('start_index: ' + str(start_i))
            except Exception, e:
                print e
                start_i = 0
        else:
            start_i = 0

        while start_i < len(self.img_files_1):
            old_img = self.img_files_1[start_i]

            print('[total] %d; [index] %d; [name] %s' % (len(self.img_files_1), start_i, old_img))

            img_1 = cv2.imread(old_img)
            img_1 = cv2.resize(img_1, (img_1.shape[1], img_1.shape[0]))
            cv2.imshow('old_image', img_1)

            img_id = old_img.split(os.sep)[-1].split('_')[0]
            found_imgs = [img_path for img_path in self.img_files_2 if img_id + "_" in img_path]
            print(found_imgs)

            for i, img_path in enumerate(found_imgs):
                img_2 = cv2.imread(img_path)
                img_2 = cv2.resize(img_2, (img_2.shape[1]*times, img_2.shape[0]*times))
                cv2.imshow('image_' + str(i), img_2)

            while True:
                k = cv2.waitKey(1) & 0xFF
                if k == ord('y'):
                    print('copy image ...')
                    # shutil.move(self.img_files_2[start_i], self.output_dir)
                    shutil.copy(old_img, self.output_dir)
                    start_i += 1
                    common.write_data(self.index_file, str(start_i), 'w')
                    break

                if k == ord('n'):
                    start_i += 1
                    common.write_data(self.index_file, str(start_i), 'w')
                    break

            # cv2.destroyAllWindows()


if __name__ == '__main__':
    # image_dir_1 = "../capture_image/province_nosign/failed_recognize/"
    # image_dir_2 = "../capture_image/province_nosign/failed_recognize_routh/"
    # image_dir_1 = "../capture_image/province_nosign/failed_detect/"
    # image_dir_2 = "../capture_image/province_nosign/failed_detect_routh/"

    # image_dir_1 = "../capture_image/998385_no_0116_multi/failed_recognize/"
    # image_dir_2 = "../capture_image/998385_no_0116_multi/failed_recognize_routh/"
    image_dir_1 = "../capture_image/997736_no_0131_one/failed_detect/"
    image_dir_2 = "../capture_image/997736_no_0131_one/failed_detect_routh/"

    output_dir = "./Data/"

    index_file = "./index.txt"
    filter_img = FilterImage(image_dir_1, image_dir_2, output_dir, index_file)

    filter_img.filter_start()
