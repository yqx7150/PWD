import json
import os
import shutil
from functools import reduce
from typing import Union, Iterable

import cv2
import pydicom
import numpy as np
from numpy import ndarray
from scipy import signal


class XDataset:
    def __init__(self, path: str):
        self._path = path
        self.help_file = os.path.join(path, 'help.json')

    def __iter__(self):
        for entry in os.listdir(self._path):
            entry_path = os.path.join(self._path, entry)
            xsd_path = os.path.join(entry_path, 'xsd.json')
            if os.path.isfile(xsd_path):
                with open(xsd_path, 'r') as xsd_file:
                    yield json.load(xsd_file)

    def update_tag_help(self, tag: str, help_info: str):
        help = {}
        if os.path.isfile(self.help_file):
            with open(self.help_file, 'r') as help_file:
                help = json.load(help_file)
        help[tag] = help_info
        with open(self.help_file, 'w') as help_file:
            json.dump(help, help_file)

    def delete_help(self, tag: str):
        if os.path.isfile(self.help_file):
            with open(self.help_file, 'r') as help_file:
                help = json.load(help_file)
        if tag in help:
            help.pop(tag)
        with open(self.help_file, 'w') as help_file:
            json.dump(help, help_file)

    def get_help(self, tag: Union[str, None]):
        help = {}
        if os.path.isfile(self.help_file):
            with open(self.help_file, 'r') as help_file:
                help = json.load(help_file)

        if tag is None:
            return help
        elif tag in help:
            return help[tag]
        else:
            return 'There is no information about {}'.format(tag)

    def select(self, selector):
        return list(filter(selector, self))

    def import_raw_files(self, path: str, entry_uid=''):
        cfg_path = ''
        raw_path = ''
        for dir_path, dir_names, file_names in os.walk(path):
            if 'fdkPara.xml' in file_names:
                cfg_path = dir_path
            if file_names and file_names[0].endswith('.raw'):
                raw_path = dir_path
                if entry_uid == '':
                    entry_uid = os.path.split(os.path.split(dir_path)[0])[1]
            if entry_uid and cfg_path and raw_path:
                scan_mode = os.path.split(cfg_path)[1]
                entry_path = os.path.join(self._path, entry_uid)
                if not os.path.exists(entry_path):
                    os.makedirs(entry_path)
                    shutil.copytree(cfg_path, os.path.join(entry_path, scan_mode))
                    shutil.copytree(raw_path, os.path.join(entry_path, 'raw'))
                    item = {
                        'uid': entry_uid,
                        'cfg_path': scan_mode,
                        'scan_mode': scan_mode,
                        'raw_path': 'raw'
                    }
                    xsd_path = os.path.join(self._path, entry_uid, 'xsd.json')
                    with open(xsd_path, 'w') as xsd_file:
                        json.dump(item, xsd_file)
                else:
                    print('exist uid: {}'.format(entry_uid))
                entry_uid = ''
                cfg_path = ''
                raw_path = ''

    def import_dicom_dirs(self, path: str, tags: dict = None):
        for dir_path, dir_names, file_names in os.walk(path):
            if file_names and file_names[0].endswith('.dcm'):
                dcm1 = pydicom.read_file(os.path.join(dir_path, file_names[0]))
                entry_uid = dcm1.SeriesInstanceUID
                if entry_uid.endswith('.0'):
                    entry_uid = entry_uid[:-2]
                if self.get_tag(entry_uid, 'dicom') is not None:
                    print("{} dicom exist".format(entry_uid))
                else:
                    self.save_dir(entry_uid, 'dicom', dir_path)
                    self.set_tag(entry_uid, 'pid', dcm1.PatientID)
                    if dict is not None:
                        for key in tags.keys():
                            self.set_tag(entry_uid, key, tags[key])
                    print("import {} dicom".format(entry_uid))

    def load_from_dicom(self, uid: str):
        dcm_path = self.path(uid, 'dicom')
        if not os.path.exists(dcm_path):
            print("{} dicom file not exists".format(uid))
        z_num = len(list(filter(lambda x: x.endswith('.dcm'), os.listdir(dcm_path))))
        dcm_data = pydicom.read_file(os.path.join(dcm_path, '{:0>5d}.dcm'.format(1)))


        data1 = dcm_data.pixel_array
        is_flip1 = int(dcm_data.ImageOrientationPatient[0]) > 0
        is_flip2 = int(dcm_data.ImageOrientationPatient[4]) > 0
        vol = np.zeros((z_num, data1.shape[0], data1.shape[1]), dtype=np.float32)
        for i in range(z_num):
            dcm_data = pydicom.read_file(os.path.join(dcm_path, '{:0>5d}.dcm'.format(i + 1)))
            data = dcm_data.pixel_array
            if is_flip1:
                data = np.flip(data, axis=0)
            if is_flip2:
                data = np.flip(data, axis=1)
            vol[z_num - i - 1, :, :] = data / 5000 + 0.2
        return vol

    def update(self, item: dict[str, str]):
        entry_path = os.path.join(self._path, item['uid'])
        xsd_path = os.path.join(entry_path, 'xsd.json')
        with open(xsd_path, 'w') as xsd_file:
            json.dump(item, xsd_file)

    def set_tag(self, uid: str, key: str, value: str):
        entry_path = os.path.join(self._path, uid)
        xsd_path = os.path.join(entry_path, 'xsd.json')
        if os.path.isfile(xsd_path):
            with open(xsd_path, 'r') as xsd_file:
                item = json.load(xsd_file)
        else:
            item = {'uid': uid, key: value}
        with open(xsd_path, 'w') as xsd_file:
            item[key] = value
            json.dump(item, xsd_file)

    def get_tag(self, uid: str, key: str):
        entry_path = os.path.join(self._path, uid)
        xsd_path = os.path.join(entry_path, 'xsd.json')
        if os.path.isfile(xsd_path):
            with open(xsd_path, 'r') as xsd_file:
                item = json.load(xsd_file)
                if key in item:
                    return item[key]
                else:
                    return None
        return None

    def delete_tag_for_all(self, key: str):
        infos = self.select(lambda x: key in x)
        for info in infos:
            uid = info['uid']
            self.delete_tag(uid, key)
        self.delete_help(key)

    def delete_tag(self, uid: str, key: str):
        entry_path = os.path.join(self._path, uid)
        xsd_path = os.path.join(entry_path, 'xsd.json')
        if os.path.isfile(xsd_path):
            with open(xsd_path, 'r') as xsd_file:
                item = json.load(xsd_file)
            tag_path = self.path(uid, key)
            if os.path.isfile(tag_path):
                os.remove(tag_path)
            if os.path.isdir(tag_path):
                shutil.rmtree(tag_path)
            item.pop(key)
            with open(xsd_path, 'w') as xsd_file:
                json.dump(item, xsd_file)

    def tag_rename_for_all(self, tag: str, new_name: str):
        infos = self.select(lambda x: tag in x)
        for info in infos:
            uid = info['uid']
            self.tag_rename(uid, tag, new_name)
        self.delete_help(tag)

    def tag_rename(self, uid: str, tag: str, new_name: str):
        entry_path = os.path.join(self._path, uid)
        xsd_path = os.path.join(entry_path, 'xsd.json')
        if os.path.isfile(xsd_path):
            with open(xsd_path, 'r') as xsd_file:
                item = json.load(xsd_file)
            tag_path = self.path(uid, tag)
            if os.path.isfile(tag_path) or os.path.isdir(tag_path):
                item[new_name] = new_name
                new_path = os.path.join(self._path, uid, new_name)
                os.rename(tag_path, new_path)
            else:
                item[new_name] = item[tag]
            item.pop(tag)
            with open(xsd_path, 'w') as xsd_file:
                json.dump(item, xsd_file)

    def save_bin(self, uid: str, name: str, binary_data: Union[ndarray, bytes], file_type=''):
        dir_path = os.path.join(self._path, uid)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        path = os.path.join(self._path, uid, name + file_type)
        with open(path, 'wb') as fout:
            binary_data.tofile(fout)
            self.set_tag(uid, name, str(binary_data.shape) + ':' + str(binary_data.dtype))

    def read_bin(self, uid: str, name: str, file_type='') -> ndarray:
        path = os.path.join(self._path, uid, name + file_type)
        formats = self.get_tag(uid, name)
        dtype = formats.split(':')[1]
        shape = [int(x) for x in str(formats.split(':')[0]).strip(")(").split(',')]
        return np.fromfile(path, dtype).reshape(shape)

    def save_file(self, uid: str, name: str, src_path: str, file_name=None):
        if file_name is None:
            file_name = name
        path = os.path.join(self._path, uid, file_name)
        shutil.copyfile(src_path, path)
        self.set_tag(uid, name, file_name)

    def save_file_to_tag(self, uid: str, name: str, src_path: str, file_name=None):
        tag_path = os.path.join(self._path, uid, name)
        if file_name is None:
            file_name = src_path.split(os.sep)[-1]
        if not os.path.exists(tag_path):
            os.makedirs(tag_path)
            self.set_tag(uid, name, name)
        shutil.copyfile(src_path, os.path.join(tag_path, file_name))

    def save_dir(self, uid: str, name: str, dir_path: str):
        path = os.path.join(self._path, uid, name)
        if not os.path.exists(path):
            shutil.copytree(dir_path, path)
            self.set_tag(uid, name, name)

    def path(self, uid: str, name: str):
        return os.path.join(self._path, uid, self.get_tag(uid, name))

    def exist(self, uid: str, name: str) -> bool:
        path = os.path.join(self._path, uid, name)
        return os.path.exists(path)

    def save(self, uid: str, name: str, data):
        if data is ndarray:
            self.save_bin(uid, name, data)
        if data is str:
            self.save_file(uid, name, data)

    def delete(self, uid: str):
        path = os.path.join(self._path, uid)
        if os.path.exists(path):
            shutil.rmtree(path)
            print("delete: {} succeed".format(uid))
        else:
            print("no uid = {}".format(uid))


def sharpen(src, ratio):
    sharpen_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    src_sharpen = signal.convolve2d(src, sharpen_kernel, mode='same')
    return src + ratio * src_sharpen


if __name__ == '__main__':
    db = XDataset(r'/data/disk6/ai_data')
    # infos = xds_c.select('dicom')
    # for i in infos:
    #     if db.get_tag(i['uid'], 'dicom') is None:
    #         print(i['uid'])

    # print(db.get_help('gen_panor_temp'))
    # db.delete_tag_for_all('gen_panor_temp')
    # db.delete_tag_for_all('auto_gen_panor')
    # for i in range(3, 8):
    db.import_dicom_dirs('/data/disk5/售后展示数据',
                         {'hospital': "售后展示数据", 'save_time': '20231010'})

    # data_path = '/data/disk5/东莞好佰年南城店硬盘拷贝_sp/5'
    # out_path = "/data/disk5/东莞好佰年南城店硬盘拷贝_sp/5"
    # dir_names = os.listdir(data_path)
    # dir_names.sort()
    # for i in range(len(dir_names)):
    #     out_ = os.path.join(out_path, "4_" + str(i // 100))
    #     if not os.path.exists(out_):
    #         os.makedirs(out_)
    #     shutil.move(os.path.join(data_path, dir_names[i]), os.path.join(out_, dir_names[i]))
    #     print(i)
    # src_path = r'/data/disk5/panor_jpg/gen_ai_data'
    # for root, dirs, files in os.walk(src_path):
    #     for file in files:
    #         if file.endswith('jpg'):
    #             img = cv2.imread(os.path.join(root, file), 0)
    #             print(os.path.join(root, file))
    #             img = sharpen(img, 0.5)
    #             cv2.imwrite(os.path.join(root, file), img)
