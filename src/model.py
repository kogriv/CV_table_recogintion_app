from ocr_engine import table_extractor
Slices_Folder = "/tmp/slices_folder"
#is_scan = 'not scan'


def process_image(file: str, is_scan='not scan'):
    """
    Process image, extracts data and returns result
    :param file:  path to the image to process
    :return:
    """
    slices_folder=Slices_Folder
    # Замените pic_type и slices_folder на соответствующие значения
    extractor = table_extractor(is_scan, file, slices_folder)
    extractor.extract()

    return {"result": [("12-01-2031", "Text", "Info"), ("15-02-2031", "Text2", "Info2")]}
