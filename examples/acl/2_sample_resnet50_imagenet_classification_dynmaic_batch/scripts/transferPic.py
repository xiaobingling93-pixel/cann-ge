# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import numpy as np
import os
from PIL import Image

def process(input_path):
    try:
        input_image = Image.open(input_path)
        input_image = input_image.resize((256, 256))
        # hwc
        img = np.array(input_image)
        height = img.shape[0]
        width = img.shape[1]
        h_off = int((height-224)/2)
        w_off = int((width-224)/2)
        crop_img = img[h_off:height-h_off, w_off:width-w_off, :]
        img = crop_img[:, :, :]
        shape = img.shape
        img = img.astype("float32")
        img[:, :, 0] /= 255
        img[:, :, 1] /= 255
        img[:, :, 2] /= 255
        img = img.reshape([1] + list(shape))
        result = img.transpose([0, 3, 1, 2])
        output_name = input_path.split('.')[0] + ".bin"
        result.tofile(output_name)
    except Exception as except_err:
        print(except_err)
        return 1
    else:
        return 0
if __name__ == "__main__":
    count_ok = 0
    count_ng = 0
    images = os.listdir(r'./')
    for image_name in images:
        if not image_name.endswith("jpg"):
            continue
        print("start to process image {}....".format(image_name))
        ret = process(image_name)
        if ret == 0:
            print("process image {} successfully".format(image_name))
            count_ok = count_ok + 1
        elif ret == 1:
            print("failed to process image {}".format(image_name))
            count_ng = count_ng + 1
    if count_ng == 0:
        print("{} images in total, {} images process successfully" .format(count_ok + count_ng, count_ok))
    else:
        print("{} images in total, {} images process successfully, {} images process failed"
            .format(count_ok + count_ng, count_ok, count_ng))
