
## Задача проекта

Задачей нашего проекта является создание наиболее универсального и эффективного решения для оцифровке таблицы с данными о донации крови с непредобработанной фотографии листа A4.

Пример такой фотографии, на основе которой далее будет объясняться решение:

![213950 (1)](https://github.com/kogriv/droo/assets/124534158/6639651d-9f10-4bc0-872a-9f1ceca5d35e)


## Предобработка изображения

Наша работа началась с предобработки изображения, а именно с распознания на листе А4 таблицы с данными о донациях и дальнейшей ее подготовки.

Для работы с непредобработанными изображениями, с наклоном, неидеальной яркостью и немного мятыми, подходит далеко не каждая заранее обученная модель, именно поэтому данным этап представляет из меня достаточно комплексное решение.

### Распознание таблицы

На первом этапе нами была протестированна модель от Microsoft. Она достаточно плохо показывала себя с некачественными фотографиями, поэтому мы от нее отказались.

```ruby

file_path = 'your path'
image = Image.open(file_path).convert("RGB")

image_processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")
model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")

inputs = image_processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# convert outputs (bounding boxes and class logits) to COCO API
target_sizes = torch.tensor([image.size[::-1]])
results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[
    0
]

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
        f"Detected {model.config.id2label[label.item()]} with confidence "
        f"{round(score.item(), 3)} at location {box}"
    )
```

В ходе тестирования нами было найдена самая подходящая модель, а именно __TahaDouaji/detr-doc-table-detection__, которая строится на dert-resnet-50 и поэтому хорошо справляется с фотографиями нашей задачи.

```ruby
image = Image.open(file_path).convert("RGB")

processor = DetrImageProcessor.from_pretrained("TahaDouaji/detr-doc-table-detection")
model = DetrForObjectDetection.from_pretrained("TahaDouaji/detr-doc-table-detection")

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
    )
```

В результате мы получаем __координаты таблицы__, относительно верхнего левого угла фотографии в формате [__x, y__ - верхний левый угол таблицы, __x, y__ - нижний правый угол таблицы]

Иногда не хватает буквально пару пикселей для того, что таблица распозналась четко. Это случается из-за наклона изображения и помятостей на нем. Поэтому было принято решение сделать координаты более универсальными. 

Мы перестали обрезать нижную часть таблицы, а от верхних координат отняли несколько пикселей, что помогло получить нам полноценную таблицу.

```ruby
orig_size = list(image.size)
box[3] = orig_size[1]
box[1] = box[1] - 75
box[0] = 0
box[2] = orig_size[0]
```
![Без названия](https://github.com/kogriv/droo/assets/124534158/b49ecf40-ddb7-4e72-b329-bbbfb7591b6a)

Далее перед нами стояла задача оставить на фотографии только таблицу. Для этого мы использовали библиотеку OpenCV.

__Далее результат предобработки будет показываться на изначальном изображении для большего понимания процесса__

Изначально картинка была бинаризирована, белый и черный цвет поменены местами, чтобы упростить распознания контуров, и были отолщены контуры таблицы и текста также для более простого распознания в будущем.

```ruby
np_image = np.asarray(image)
grayscale_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
thresholded_image = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
inverted_image = cv2.bitwise_not(thresholded_image)
dilated_image = cv2.dilate(inverted_image, None, iterations=5)
```

Затем нами были найдены любые формы на изображении (контуры, текст и прочие).

```ruby
contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
image_with_all_contours = np_image.copy()
cv2.drawContours(image_with_all_contours, contours, -1, (0, 255, 0), 3)
```
Результат:
![Без названия (1)](https://github.com/kogriv/droo/assets/124534158/7084184d-917c-41dd-904c-95e933de2da4)

Далее мы избавились от веделения текста, оставив только контуры ячеек таблицы.
```ruby
rectangular_contours = []
for contour in contours:
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    if len(approx) == 4:
        rectangular_contours.append(approx)

image_with_only_rectangular_contours = np_image.copy()
cv2.drawContours(image_with_only_rectangular_contours, rectangular_contours, -1, (0, 255, 0), 3)
```
![Без названия (2)](https://github.com/kogriv/droo/assets/124534158/c010211d-2ea4-43bc-b178-7b96ccb0e2ac)


Благодаря этому мы смогли найти самый большой прямоугольник, который является контуром всей таблицы.

```ruby
max_area = 0
contour_with_max_area = None
for contour in rectangular_contours:
    area = cv2.contourArea(contour)
    if area > max_area:
        max_area = area
        contour_with_max_area = contour

image_with_contour_with_max_area = np_image.copy()
cv2.drawContours(image_with_contour_with_max_area, [contour_with_max_area], -1, (0, 255, 0), 3)

```
![Без названия (3)](https://github.com/kogriv/droo/assets/124534158/fa9e3bf2-a224-4284-8e54-dc6edb9f70aa)

На данном этапе мы получили четко выделенную таблицу, теперь осталось только убрать все лишнее вокруг нее.
Для выполнения этой задачи мы нашли края полученного контура.

```ruby
def order_points(pts):

        pts = pts.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect


contour_with_max_area_ordered = order_points(contour_with_max_area)
image_with_points_plotted = np_image.copy()
for point in contour_with_max_area_ordered:
        point_coordinates = (int(point[0]), int(point[1]))
        image_with_points_plotted = cv2.circle(image_with_points_plotted, point_coordinates, 10, (0, 0, 255), -1)
cv2_imshow(image_with_points_plotted)
```
![Без названия (4)](https://github.com/kogriv/droo/assets/124534158/35a52cdf-ed30-4112-9db3-dcce7f4262b1)

И повернули таблицу, обрезав все лишнее, получив идеально преобработанное изображение.

```ruby
def calculateDistanceBetween2Points(p1, p2):
        dis = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
        return dis

existing_image_width = np_image.shape[1]
existing_image_width_reduced_by_10_percent = int(existing_image_width * 0.9)

distance_between_top_left_and_top_right = calculateDistanceBetween2Points(contour_with_max_area_ordered[0], contour_with_max_area_ordered[1])
distance_between_top_left_and_bottom_left = calculateDistanceBetween2Points(contour_with_max_area_ordered[0], contour_with_max_area_ordered[3])
aspect_ratio = distance_between_top_left_and_bottom_left / distance_between_top_left_and_top_right
new_image_width = existing_image_width_reduced_by_10_percent
new_image_height = int(new_image_width * aspect_ratio)

pts1 = np.float32(contour_with_max_area_ordered)
pts2 = np.float32([[0, 0], [new_image_width, 0], [new_image_width, new_image_height], [0, new_image_height]])
matrix = cv2.getPerspectiveTransform(pts1, pts2)
perspective_corrected_image = cv2.warpPerspective(dilated_image, matrix, (new_image_width, new_image_height))
perspective_corrected_orig_image = cv2.warpPerspective(np_image, matrix, (new_image_width, new_image_height))
cv2_imshow(perspective_corrected_orig_image)
```
![Без названия (5)](https://github.com/kogriv/droo/assets/124534158/f02a5116-c594-41e2-884e-740af3bd24d3)

## Распознание контуров таблицы

На данной этапе была поставлена задача распознать все горизонтальные и вертикальные линии таблицы, чтобы найти их координаты для четкого разграничения ячеек, и чтобы удалит их для распознания.

__Задача этапа: получить изображение без контуров таблицы__

Для решения этой задачи мы нашли горизонтальные линии таблицы, затем вертикальные и объединили их.

```ruby
hor = np.array([[1,1,1,1,1,1]])
vertical_lines_eroded_image = cv2.erode(perspective_corrected_image, hor, iterations=100)
vertical_lines_eroded_image = cv2.dilate(vertical_lines_eroded_image, hor, iterations=100)
ver = np.array([[1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1]])
horizontal_lines_eroded_image = cv2.erode(perspective_corrected_image, ver, iterations=100)
horizontal_lines_eroded_image = cv2.dilate(horizontal_lines_eroded_image, ver, iterations=100)
combined_image = cv2.add(vertical_lines_eroded_image, horizontal_lines_eroded_image)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
combined_image_dilated = cv2.dilate(combined_image, kernel, iterations=7)
```
![Без названия (6)](https://github.com/kogriv/droo/assets/124534158/65e17253-61d7-4f74-8870-9017da7200c0)

Затем мы в два этапа удалили полученные контуры:

```ruby
image_without_lines = cv2.subtract(perspective_corrected_image, combined_image_dilated)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
image_without_lines_noise_removed = cv2.erode(image_without_lines, kernel, iterations=5)
image_without_lines_noise_removed = cv2.dilate(image_without_lines_noise_removed, kernel, iterations=5)
```
![Без названия (7)](https://github.com/kogriv/droo/assets/124534158/f311a1db-c09d-4ce7-a0d2-3f8f04cac581)

## Распознание текста

Данную задачу мы решили сделать в данной последовательности:
- Распознать, где именно находится текст и получить координаты этих мест.
- Получить последовательность расположения строк и столбцов.
- Вырезать каждый элемент текста отдельно и распознать его.
- Собрать таблицу обратно на основе последовательности строк и столбцов.

Для начала мы сильно увеличили все элементы на изображении, что найти те рамки, где находится какой-либо текст.

```ruby
kernel_to_remove_gaps_between_words = np.array([
                [1,1,1,1,1,1,1,1,1,1],
               [1,1,1,1,1,1,1,1,1,1]
        ])
dilated_image = cv2.dilate(image_without_lines_noise_removed, kernel_to_remove_gaps_between_words, iterations=5)
simple_kernel = np.ones((5,5), np.uint8)
dilated_image = cv2.dilate(image_without_lines_noise_removed, simple_kernel, iterations=5)
```
![Без названия (8)](https://github.com/kogriv/droo/assets/124534158/3cf7a17e-b026-4a9c-b7d7-3503f56de5ae)


Далее мы нашли контуры белых фигур и перенесли их на оригинальное изображении. Контуры были приведены к прямоугольному формату, что помогло распознавать текст значительно лучше.

```ruby
result = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = result[0]
image_with_contours_drawn = perspective_corrected_orig_image.copy()
cv2.drawContours(image_with_contours_drawn, contours, -1, (0, 255, 0), 3)

approximated_contours = []
for contour in contours:
    approx = cv2.approxPolyDP(contour, 3, True)
    approximated_contours.append(approx)

image_with_contours = perspective_corrected_orig_image.copy()
cv2.drawContours(image_with_contours, approximated_contours, -1, (0, 255, 0), 5)

bounding_boxes = []
image_with_all_bounding_boxes = perspective_corrected_orig_image.copy()
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    bounding_boxes.append((x, y, w, h))
    image_with_all_bounding_boxes = cv2.rectangle(image_with_all_bounding_boxes, (x, y), (x + w, y + h), (0, 255, 0), 3)
```

![Без названия (9)](https://github.com/kogriv/droo/assets/124534158/efa7c4a5-02b3-400b-8b0c-11da4af8ff62)

Далее мы получили представление расположения ячеек относительно друг друга:

```ruby
def get_mean_height_of_bounding_boxes():
    heights = []
    for bounding_box in bounding_boxes:
        x, y, w, h = bounding_box
        heights.append(h)
    return np.mean(heights)
bounding_boxes = sorted(bounding_boxes, key=lambda x: x[1])
mean_height = get_mean_height_of_bounding_boxes()

rows = []
half_of_mean_height = mean_height / 2
current_row = [ bounding_boxes[0] ]
for bounding_box in bounding_boxes[1:]:
    current_bounding_box_y = bounding_box[1]
    previous_bounding_box_y = current_row[-1][1]
    distance_between_bounding_boxes = abs(current_bounding_box_y - previous_bounding_box_y)
    if distance_between_bounding_boxes <= half_of_mean_height:
        current_row.append(bounding_box)
    else:
        rows.append(current_row)
        current_row = [ bounding_box ]
rows.append(current_row)
for row in rows:
            row.sort(key=lambda x: x[0])
```

Далее самое интересное. По полученным координатам мы обрезаем каждую ячейку и с помощью Tesseract-OCR распознаем в ней текст. Затем располагаем все в матрицу в том же порядке, что и взяли.

Настройки Tesseract:
```ruby
 - -l rus --oem 3 --psm 7 --dpi 72 -c tessedit_char_whitelist="йцукенгшщзхъфывапролджэячсмитьбю/ЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭЯЧСМИТЬБЮ0123456789().calmg* 
```

```ruby
table = []
current_row = []
image_number = 0
for row in rows:
    for bounding_box in row:
        x, y, w, h = bounding_box
        y = y - 5
        cropped_image = perspective_corrected_orig_image[y:y+h, x:x+w]
        image_slice_path = "./ocr_slices/img_" + str(image_number) + ".jpg"
        cv2.imwrite(image_slice_path, cropped_image)
        results_from_ocr = get_result_from_tersseract(image_slice_path)
        current_row.append(results_from_ocr)
        image_number += 1
    table.append(current_row)
    current_row = []
```

На выходе мы получаем неотредактированную таблицу с данными:

![Screenshot_592](https://github.com/kogriv/droo/assets/124534158/168a8817-a2fc-41a2-8892-44b308b8e5a7)


## Форматирование результата распознания

