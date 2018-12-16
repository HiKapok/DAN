# Copyright 2018 Changan Wang

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
import cv2

import sys; sys.path.insert(0, ".")

def colors_subselect(colors, num_classes=2):
    dt = len(colors) // num_classes
    sub_colors = []
    for i in range(num_classes):
        color = colors[i*dt]
        if isinstance(color[0], float):
            sub_colors.append([int(c * 255) for c in color])
        else:
            sub_colors.append([c for c in color])
    return sub_colors

import matplotlib.cm as mpcm
colors = colors_subselect(mpcm.plasma.colors, num_classes=26)
print(colors)
#colors_tableau = [[12, 7, 134], [64, 3, 156], [104, 0, 167], [142, 12, 164], [174, 39, 145], [200, 68, 122], [222, 96, 100], [239, 126, 78], [250, 160, 57], [253, 198, 38]]
colors_tableau = [(12, 7, 134), (203, 71, 119)]

# ymin, xmin, ymax, xmax
def bboxes_draw_on_img(img, classes, scores, bboxes, thickness=2, y_first=False):
    shape = img.shape
    scale = 0.4
    text_thickness = 1
    line_type = 8
    for i in range(bboxes.shape[0]):
        if classes[i] < 1: continue
        bbox = bboxes[i]
        color = colors_tableau[classes[i]]
        # Draw bounding boxes
        if y_first:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[2]), int(bbox[3]))
        else:
            p1 = (int(bbox[1]), int(bbox[0]))
            p2 = (int(bbox[3]), int(bbox[2]))
        if (p2[0] - p1[0] < 0) or (p2[1] - p1[1] < 0):
            continue

        cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)
        # Draw text
        s = '%.1f%%' % (scores[i] * 100)
        # text_size is (width, height)
        text_size, baseline = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, text_thickness)
        p1 = (p1[0] - text_size[1], p1[1])

        cv2.rectangle(img, (p1[1] - thickness//2, p1[0] - thickness - baseline), (p1[1] + text_size[0], p1[0] + text_size[1]), color, -1)

        cv2.putText(img, s, (p1[1], p1[0] + baseline), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), text_thickness, line_type)

    return img

def bboxes_gt_draw_on_img(img, classes, scores, bboxes, gts, thickness=2):
    shape = img.shape
    scale = 0.4
    text_thickness = 1
    line_type = 8
    for i in range(bboxes.shape[0]):
        if classes[i] < 1: continue
        bbox = bboxes[i]
        color = colors_tableau[classes[i]]
        # Draw bounding boxes
        p1 = (int(bbox[1]), int(bbox[0]))
        p2 = (int(bbox[3]), int(bbox[2]))
        if (p2[0] - p1[0] < 0) or (p2[1] - p1[1] < 0):
            continue

        cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)
        # Draw text
        s = '%.1f%%' % (scores[i] * 100)
        # text_size is (width, height)
        text_size, baseline = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, text_thickness)
        p1 = (p1[0] - text_size[1], p1[1])

        cv2.rectangle(img, (p1[1] - thickness // 2, p1[0] - thickness - baseline), (p1[1] + text_size[0], p1[0] + text_size[1]), color, -1)

        cv2.putText(img, s, (p1[1], p1[0] + baseline), cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), text_thickness, line_type)
    for i in range(gts.shape[0]):
        bbox = gts[i]
        color = colors_tableau[0]
        # Draw bounding boxes
        p1 = (int(bbox[1]), int(bbox[0]))
        p2 = (int(bbox[3]), int(bbox[2]))
        if (p2[0] - p1[0] < 0) or (p2[1] - p1[1] < 0):
            continue

        cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)
    return img


# xmin, ymin, xmax, ymax
def absolute_bboxes_draw_on_img(img, classes, scores, bboxes, thickness=2):
    shape = img.shape
    scale = 0.4
    text_thickness = 1
    line_type = 8
    all_objs = 0
    for i in range(bboxes.shape[0]):
        if classes[i] < 1: continue
        bbox = bboxes[i]
        color = colors_tableau[classes[i]]
        # Draw bounding boxes
        p1 = (int(bbox[1]), int(bbox[0]))
        p2 = (int(bbox[3]), int(bbox[2]))
        if (p2[0] - p1[0] < 0) or (p2[1] - p1[1] < 0):
            continue

        all_objs += 1

        cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)
        # Draw text
        s = '%.1f%%' % (scores[i] * 100)
        # text_size is (width, height)
        text_size, baseline = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, text_thickness)
        p1 = (p1[0] - text_size[1], p1[1])

        #cv2.rectangle(img, (p1[1] - thickness//2, p1[0] - thickness - baseline), (p1[1] + text_size[0], p1[0] + text_size[1]), color, -1)

        #cv2.putText(img, s, (p1[1], p1[0] + baseline), cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), text_thickness, line_type)
    print("all instances: {}".format(all_objs))
    return img

