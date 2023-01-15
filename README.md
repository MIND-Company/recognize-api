Итак, инструкция по установке модели:
1. Скачать этот репозиторий
2. Установить зависимости из корневой папки
3. Установить зависимости /yolo/requirements.txt
4. Скачать Tesseract OCR (я брал тут вроде: https://github.com/UB-Mannheim/tesseract/wiki, для Линукса мб можно проще поставить)
5. В файле recognition_model.py поменять путь до tesseract на свой (pytesseract.pytesseract.tesseract_cmd = ...)
6. Запустить файл main.py
Ну и вроде бы все :)