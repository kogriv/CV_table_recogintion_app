# Создание докер образов для решения задачи распознавания таблиц

Поскольку требуемые для задачи пакеты занимают большой объем и с целью избежания длительной пересборки образа были созданы 2 докерфайла:
- Dockerfile.stable - создание базового образа, содержащего только необходимые програмные пакеты (requirements_stable.txt). Созданный образ - ocr_5_base:1.0, находится в репозитории https://hub.docker.com/repository/docker/kogriv/ocr_5_base/general
- Dockerfile.final - создание образа на основе базового ocr_5_base:1.0. Добавляются файлы самой модели и приложения, при необходимости добавьте в список зависимостей (requirements_new.txt) щепотку каких то дополнительных пакетов по вкусу. Созданные образ ocr_5_model:1.0 - расположен в репозитории https://hub.docker.com/repository/docker/kogriv/ocr_5_model/general

## Создание и запуcк контейнера:

docker build -t ocr_5_base:1.0 -f Dockerfile.stable .

docker build -t ocr_5_model:1.0 -f Dockerfile.final .

docker run -d -p 8010:8000 --name ocr ocr_5_model:1.0

## Пример подключения к сервису приложения в контейнере

```python
# Установите URL-адрес веб-сервера
url = 'http://localhost:8010/ocr'

# Установите путь к файлу изображения
# image_path = 'C:/0/405/s/204119.jpg'
image_file = 'C:/0/405/s/256838.jpg'
filename, _ = os.path.splitext(os.path.basename(image_file))
csv_filename = f'{filename}.csv'

# выбор скан / не скан : scan / not scan
url_with_params = f"{url}?is_scan=not scan"

# Открываем файл и отправляем запрос с помощью multipart/form-data
with open(image_file, 'rb') as file:
    files = {'file': file}
    response = requests.post(url_with_params, files=files)

# вывод ответа от сервиса
print(response.text)

# Запись CSV файла по указанному адресу
output_dir = r'C:\0\405\output'
# Проверяем статус ответа
if response.status_code == 200:
    # Сохраняем CSV файл в директорию output_dir
    csv_file = os.path.join(output_dir, csv_filename)
    
    # Сохраняем файл
    with open(csv_file, 'wb') as file:
        file.write(response.content)
    
    print(f'CSV файл сохранен: {csv_file}')
else:
    print('Ошибка при получении ответа от сервера')
```
