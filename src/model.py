from ocr_engine import table_extractor
import os
Slices_Folder = "/tmp/slices_folder"
Result_Folder = "/tmp/result_folder"
#is_scan = 'not scan'


def process_image(file: str
                  , is_scan
                  ):
    """
    Process image, extracts data and returns result
    :param file:  path to the image to process
    :return:
    """
    slices_folder=Slices_Folder
    result_folder=Result_Folder
    # Замените pic_type и slices_folder на соответствующие значения
    extractor = table_extractor(is_scan, file, slices_folder,result_folder)
    extractor.extract()
    extractor.create_csv_table()

    base_filename = os.path.basename(file)
    filename, _ = os.path.splitext(base_filename)
    csv_filename = f"{filename}.csv"
    csv_path = os.path.join(extractor.result_folder, csv_filename)

    with open(csv_path, 'r', encoding='utf-8') as file:
        contents = file.read()

    return {csv_filename: contents}
