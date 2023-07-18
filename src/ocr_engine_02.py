
import cv2
import subprocess
import pandas as pd
import pytesseract
import numpy as np
import torch
from transformers import (
    AutoImageProcessor,
    DetrImageProcessor,
    DetrForObjectDetection
    )
from PIL import Image
import csv
import os


class table_extractor:

    def __init__(self, pic_type, file_path, slices_folder, result_folder):
        self.pic_type = pic_type
        self.file_path = file_path
        self.slices_folder = slices_folder
        self.result_folder = result_folder
        self.rows = []
        #self.box = None  # Добавленная строка

    def extract(self):
        self.read_image()
        self.table_area()
        self.box_fix()
        self.img_crop()
        self.dilate_image()
        self.find_table()
        self.rotate_table()
        self.delete_lines()
        self.b_boxes()
        self.get_table()
        self.raw_table_filter(self.table)
        self.postprocessing()
        self.reshaped_table = self.reshape(self.new_table, self.filtered_table)
#        self.to_csv(self.reshaped_table)

    def order_points(self, pts):
            pts = pts.reshape(4, 2)
            rect = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]
            return rect

    def calculateDistanceBetween2Points(self, p1, p2):
            dis = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
            return dis

    def get_mean_height_of_bounding_boxes(self):
        heights = []
        for bounding_box in self.bounding_boxes:
            x, y, w, h = bounding_box
            heights.append(h)
        return np.mean(heights)

    def get_result_from_tersseract(self, image_path):
            output = subprocess.getoutput('tesseract ' + image_path + ' - -l rus --oem 3 --psm 7 --dpi 72 -c tessedit_char_whitelist="йцукенгшщзхъфывапролджэячсмитьбю/ЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭЯЧСМИТЬБЮ0123456789().calmg* "')
            output = output.strip()
            return output

    def read_image(self):
        self.image = Image.open(self.file_path).convert("RGB")

    def table_area(self):
        self.processor = DetrImageProcessor.from_pretrained('TahaDouaji/detr-doc-table-detection')
        self.model = DetrForObjectDetection.from_pretrained('TahaDouaji/detr-doc-table-detection')

        self.inputs = self.processor(images=self.image, return_tensors="pt")
        self.outputs = self.model(**self.inputs)

        # конвертим outputs (ограничивающие рамки и логиты классов) к COCO API
        # оставим только обнаружения с оценкой > 0.9
        self.target_sizes = torch.tensor([self.image.size[::-1]])
        self.results = self.processor.post_process_object_detection(self.outputs, target_sizes=self.target_sizes, threshold=0.8)[0]

        for score, label, self.box in zip(self.results["scores"], self.results["labels"], self.results["boxes"]):
            self.box = [round(i, 2) for i in self.box.tolist()]
            print(
                    f"Detected {self.model.config.id2label[label.item()]} with confidence "
                    f"{round(score.item(), 3)} at location {self.box}"
            )

    def box_fix(self):
        self.orig_size = list(self.image.size)
        self.box[3] = self.orig_size[1]
        self.box[1] = self.box[1] - 75
        self.box[0] = 0
        self.box[2] = self.orig_size[0]

    def img_crop(self):
        self.image = self.image.crop(self.box)

    def dilate_image(self):
        self.np_image = np.asarray(self.image)

        if self.pic_type == 'not scan':
            self.grayscale_image = cv2.cvtColor(self.np_image, cv2.COLOR_BGR2GRAY)
            self.thresholded_image = cv2.threshold(self.grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            self.inverted_image = cv2.bitwise_not(self.thresholded_image)
            self.dilated_image = cv2.dilate(self.inverted_image, None, iterations=5)

        elif self.pic_type == 'scan':
            self.grayscale_image = cv2.cvtColor(self.np_image, cv2.COLOR_BGR2GRAY)
            self.thresholded_image = cv2.threshold(self.grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            self.inverted_image = cv2.bitwise_not(self.thresholded_image)
            self.dilated_image = cv2.dilate(self.inverted_image, None, iterations=1)

    def find_table(self):
        self.contours, self.hierarchy = cv2.findContours(self.dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.image_with_all_contours = self.np_image.copy()
        cv2.drawContours(self.image_with_all_contours, self.contours, -1, (0, 255, 0), 3)

        self.rectangular_contours = []
        for contour in self.contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:
                self.rectangular_contours.append(approx)

        self.image_with_only_rectangular_contours = self.np_image.copy()
        cv2.drawContours(self.image_with_only_rectangular_contours, self.rectangular_contours, -1, (0, 255, 0), 3)

        max_area = 0
        self.contour_with_max_area = None
        for contour in self.rectangular_contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                self.contour_with_max_area = contour

        self.image_with_contour_with_max_area = self.np_image.copy()
        cv2.drawContours(self.image_with_contour_with_max_area, [self.contour_with_max_area], -1, (0, 255, 0), 3)

    def rotate_table(self):
        self.contour_with_max_area_ordered = self.order_points(self.contour_with_max_area)
        self.image_with_points_plotted = self.np_image.copy()

        for point in self.contour_with_max_area_ordered:
            point_coordinates = (int(point[0]), int(point[1]))
            self.image_with_points_plotted = cv2.circle(self.image_with_points_plotted, point_coordinates, 10, (0, 0, 255), -1)

        existing_image_width = self.np_image.shape[1]
        existing_image_width_reduced_by_10_percent = int(existing_image_width * 0.9)

        distance_between_top_left_and_top_right = self.calculateDistanceBetween2Points(self.contour_with_max_area_ordered[0], self.contour_with_max_area_ordered[1])
        distance_between_top_left_and_bottom_left = self.calculateDistanceBetween2Points(self.contour_with_max_area_ordered[0], self.contour_with_max_area_ordered[3])

        aspect_ratio = distance_between_top_left_and_bottom_left / distance_between_top_left_and_top_right

        self.new_image_width = existing_image_width_reduced_by_10_percent
        self.new_image_height = int(self.new_image_width * aspect_ratio)

        pts1 = np.float32(self.contour_with_max_area_ordered)
        pts2 = np.float32([[0, 0], [self.new_image_width, 0], [self.new_image_width, self.new_image_height], [0, self.new_image_height]])

        matrix = cv2.getPerspectiveTransform(pts1, pts2)

        self.perspective_corrected_image = cv2.warpPerspective(self.dilated_image, matrix, (self.new_image_width, self.new_image_height))
        self.perspective_corrected_orig_image = cv2.warpPerspective(self.np_image, matrix, (self.new_image_width, self.new_image_height))

    def delete_lines(self):
        hor = np.array([[1,1,1,1,1,1]])
        self.vertical_lines_eroded_image = cv2.erode(self.perspective_corrected_image, hor, iterations=100)
        self.vertical_lines_eroded_image = cv2.dilate(self.vertical_lines_eroded_image, hor, iterations=100)

        ver = np.array(
            [[1], [1], [1],[1], [1], [1], [1]]
            )

        self.horizontal_lines_eroded_image = cv2.erode(self.perspective_corrected_image, ver, iterations=100)
        self.horizontal_lines_eroded_image = cv2.dilate(self.horizontal_lines_eroded_image, ver, iterations=100)

        if self.pic_type == 'not scan':
            self.combined_image = cv2.add(self.vertical_lines_eroded_image, self.horizontal_lines_eroded_image)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            self.combined_image_dilated = cv2.dilate(self.combined_image, kernel, iterations=7)

        elif self.pic_type == 'scan':
            self.combined_image = cv2.add(self.vertical_lines_eroded_image, self.horizontal_lines_eroded_image)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            self.combined_image_dilated = cv2.dilate(self.combined_image, kernel, iterations=5)

        self.image_without_lines = cv2.subtract(self.perspective_corrected_image, self.combined_image_dilated)

        if self.pic_type == 'not scan':
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            self.image_without_lines_noise_removed = cv2.erode(self.image_without_lines, kernel, iterations=5)
            self.image_without_lines_noise_removed = cv2.dilate(self.image_without_lines_noise_removed, kernel, iterations=5)

        elif self.pic_type == 'scan':
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            self.image_without_lines_noise_removed = cv2.erode(self.image_without_lines, kernel, iterations=2)
            self.image_without_lines_noise_removed = cv2.dilate(self.image_without_lines_noise_removed, kernel, iterations=1)

    def b_boxes(self):
        kernel_to_remove_gaps_between_words = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ])

        if self.pic_type == 'not scan':
            self.dilated_image = cv2.dilate(self.image_without_lines_noise_removed, kernel_to_remove_gaps_between_words, iterations=5)
            simple_kernel = np.ones((5, 5), np.uint8)
            self.dilated_image = cv2.dilate(self.image_without_lines_noise_removed, simple_kernel, iterations=5)
        elif self.pic_type == 'scan':
            self.dilated_image = cv2.dilate(self.image_without_lines_noise_removed, kernel_to_remove_gaps_between_words, iterations=10)
            simple_kernel = np.ones((5, 5), np.uint8)
            self.dilated_image = cv2.dilate(self.image_without_lines_noise_removed, simple_kernel, iterations=3)

        result = cv2.findContours(self.dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.contours = result[0]
        self.image_with_contours_drawn = self.perspective_corrected_orig_image.copy()
        cv2.drawContours(self.image_with_contours_drawn, self.contours, -1, (0, 255, 0), 3)

        self.approximated_contours = []
        for contour in self.contours:
            approx = cv2.approxPolyDP(contour, 3, True)
            self.approximated_contours.append(approx)

        self.image_with_contours = self.perspective_corrected_orig_image.copy()
        cv2.drawContours(self.image_with_contours, self.approximated_contours, -1, (0, 255, 0), 5)

        self.bounding_boxes = []
        self.image_with_all_bounding_boxes = self.perspective_corrected_orig_image.copy()

        if self.contours:
            for contour in self.contours:
                x, y, w, h = cv2.boundingRect(contour)
                self.bounding_boxes.append((x, y, w, h))
                self.image_with_all_bounding_boxes = cv2.rectangle(self.image_with_all_bounding_boxes, (x, y), (x + w, y + h), (0, 255, 0), 3)

            self.bounding_boxes = sorted(self.bounding_boxes, key=lambda x: x[1])
            self.mean_height = self.get_mean_height_of_bounding_boxes()
            self.rows = []

            half_of_mean_height = self.mean_height / 2
            current_row = [self.bounding_boxes[0]]

            for bounding_box in self.bounding_boxes[1:]:
                current_bounding_box_y = bounding_box[1]
                previous_bounding_box_y = current_row[-1][1]
                distance_between_bounding_boxes = abs(current_bounding_box_y - previous_bounding_box_y)
                
                if distance_between_bounding_boxes <= half_of_mean_height:
                    current_row.append(bounding_box)
                else:
                    self.rows.append(current_row)
                    current_row = [bounding_box]
            
            self.rows.append(current_row)
            for row in self.rows:
                row.sort(key=lambda x: x[0])

    def get_table(self):
        self.table = []
        current_row = []
        image_number = 0

        for row in self.rows:
            for bounding_box in row:
                x, y, w, h = bounding_box
                # y = y - 5
                cropped_image = self.perspective_corrected_orig_image[y:y+h, x:x+w]
                image_slice_path = self.slices_folder + 'img_' + str(image_number) + '.jpg'
                cv2.imwrite(image_slice_path, cropped_image)
                results_from_ocr = self.get_result_from_tersseract(image_slice_path)
                current_row.append(results_from_ocr)
                image_number += 1
            self.table.append(current_row)
            current_row = []


    def delete_redundant_elements(self, table, iter=5):

        self.filtered_table = table.copy()

        for _ in range(iter):

            for row in self.filtered_table:

                for item in row:
                    if (len(item) < 6) or (len(item) > 12):
                        row.remove(item)

        return self.filtered_table


    def get_max_row_lenght(self, table):

        self.max = 0

        for row in table:
            if len(row) > self.max:
                self.max = len(row)

        return self.max


    def delete_redundant_rows(self, table, iter=3):

        self.filtered_table = table.copy()
        self.max = self.get_max_row_lenght(self.table)

        for _ in range(iter):
            for row in self.filtered_table:
                if (len(row) <= self.max / 3) or (len(row) <= 2):
                    self.filtered_table.remove(row)

        return self.filtered_table


    def split_don_type(self, table):

        self.filtered_table = []

        for row in table:
            self.filtered_table.append(
                [splitted_item for item in row for splitted_item in item.split()]
                )

        return self.filtered_table


    def change_values(self, value: str, values: dict) -> str:
        if value in values.keys():
            return values[value]
        else:
            return value


    def raw_table_filter(self, raw_pred):
        self.filtered_table = self.delete_redundant_elements(self.table)
        self.filtered_table = self.delete_redundant_rows(self.filtered_table)
        self.filtered_table = self.split_don_type(self.filtered_table)

    def postprocessing(self):
        self.don_type = {
            'кр/д': 'Цельная кровь',
            'крид': 'Цельная кровь',
            'кри': 'Цельная кровь',
            'т/ф': 'Тромбоциты',
            'п/ф': 'Плазма',
            'пл/д': 'Плазма'
        }
        self.pay_type = {
            '(бв)': 'Безвозмездно',
            '(6в)': 'Безвозмездно',
            '(пл)': 'Платно'
        }

        self.max_len = 0
        for row in self.filtered_table:
            if len(row) > self.max_len:
                self.max_len = len(row)
        self.row_len = 3
        self.new_table = []
        for i in range(len(self.filtered_table) * int(self.max_len / 3)):
            new_row = [0 for _ in range(self.row_len)]
            self.new_table.append(new_row)
        counter = 0
        row_counter = 0
        if self.max_len == 8:
            self.max_len += 1

        for i in range(len(self.filtered_table)):
            if self.max_len == 6:
                pass
            elif self.max_len == 9 and row_counter >= len(self.new_table):
                break
            elif self.max_len == 9 and self.new_table[row_counter][2] == 0 and row_counter != 0:
                row_counter += 1
            elif self.max_len == 9 and self.new_table[row_counter].count(0) == 3:
                if row_counter % 3 == 1:
                    row_counter += 2
                elif row_counter % 3 == 2:
                    row_counter += 1
            for j in range(len(self.filtered_table[i])):
                if row_counter >= len(self.new_table):
                    break
                counter = 0
                try:
                    datetime_object = pd.to_datetime(self.filtered_table[i][j].strip('.'), format='%d.%m.%Y')
                    try:
                        if self.new_table[row_counter - 1][2] == 0 and row_counter != 0:
                            row_counter += 1
                    except:
                        pass
                    if self.new_table[row_counter][counter] != 0:
                        row_counter += 1
                    self.new_table[row_counter][counter] = self.filtered_table[i][j].strip('.')
                    continue
                except:
                    counter += 1

                if self.filtered_table[i][j] in self.don_type.keys():
                    if self.new_table[row_counter][counter] != 0:
                        row_counter += 1
                    self.new_table[row_counter][counter] = self.change_values(self.filtered_table[i][j], self.don_type)
                    continue
                else:
                    counter += 1

                if self.filtered_table[i][j] in self.pay_type.keys():
                    self.new_table[row_counter][counter] = self.change_values(self.filtered_table[i][j], self.pay_type)
                    row_counter += 1
                    continue
                else:
                    counter += 1

        self.new_table = pd.DataFrame(self.new_table, columns=['Дата донации', 'Класс крови', 'Тип донации'])

    def reshape(self, table, preds):
        if self.get_max_row_lenght(preds) == 6:
            temp_table_1 = table.iloc[::2, :]
            temp_table_2 = table.iloc[1::2, :]

            reshaped_table = pd \
                .concat([temp_table_1, temp_table_2]) \
                .reset_index(drop=True)

            return reshaped_table

        else:
            temp_table_1 = table.iloc[::3, :]
            temp_table_2 = table.iloc[1::3, :]
            temp_table_3 = table.iloc[2::3, :]

            reshaped_table = pd \
                .concat([temp_table_1, temp_table_2, temp_table_3]) \
                .reset_index(drop=True)

            return reshaped_table

    def accuracy_score(self, table_pred, table_true):

        if table_pred.shape == table_true.shape:

            rows = int(table_pred.shape[0])
            cols = int(table_pred.shape[1])
            total = rows * cols
            correct = 0
            for row in range(rows):
                for col in range(cols):
                    if table_pred.iloc[row, col] == table_true.iloc[row, col]:
                        correct += 1
                    else:
                        continue

            return correct / total

        else:
                print('Shapes of table_pred and table_true does not match!')



    def accuracy_check(self, table_pred, csv_orig_path):

            accuracy_columns = ['Дата донации', 'Класс крови', 'Тип донации']
            table_orig = pd.read_csv(csv_orig_path)

            table_pred = table_pred[accuracy_columns]
            table_orig = table_orig[accuracy_columns]

            print( self.accuracy_score(table_pred, table_orig))

    def to_csv(self, table):
        table.to_csv()

    def create_csv_table(self):
        base_filename = os.path.basename(self.file_path)
        filename, _ = os.path.splitext(base_filename)
        csv_filename = f"{filename}.csv"
        csv_path = os.path.join(self.result_folder, csv_filename)

        # Проверка на существование файла и удаление, если он уже существует
        if os.path.exists(csv_path):
            os.remove(csv_path)

        with open(csv_path, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            for row in self.table:
                writer.writerow(row)