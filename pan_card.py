import easyocr
import cv2
import matplotlib.pyplot as plt
import re
import json

class PANCardEasyOCR:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.reader = easyocr.Reader(['en'], gpu=False)
        self.results = []
        self.extracted_text = []
        self.data = {}

    def perform_ocr(self):
        temp_path = 'gray_temp.jpg'
        cv2.imwrite(temp_path, self.gray)
        self.results = self.reader.readtext(temp_path)

        for (bbox, text, prob) in self.results:
            self.extracted_text.append(text.strip())

        return self.extracted_text

    def extract_fields(self):
        full_text = ' '.join(self.extracted_text)

        # PAN Number
        pan = re.search(r'\b([A-Z]{5}[0-9]{4}[A-Z])\b', full_text)
        if pan:
            self.data['PAN Number'] = pan.group(1)

        # Name
        for i, line in enumerate(self.extracted_text):
            if "Name" in line and i + 1 < len(self.extracted_text):
                self.data['Name'] = self.extracted_text[i + 1]
                break

        # Father's Name
        for i, line in enumerate(self.extracted_text):
            if "Father" in line and i + 1 < len(self.extracted_text):
                self.data["Father's Name"] = self.extracted_text[i + 1]
                break

        # DOB
        dob = re.search(r'\d{2}/\d{2}/\d{4}', full_text)
        if dob:
            self.data['DOB'] = dob.group()

        return self.data

    def draw_boxes(self, save_path='pan_easyocr_output.jpg'):
        for (bbox, text, prob) in self.results:
            (top_left, top_right, bottom_right, bottom_left) = bbox
            top_left = tuple(map(int, top_left))
            bottom_right = tuple(map(int, bottom_right))
            cv2.rectangle(self.image, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(self.image, text, (top_left[0], top_left[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        cv2.imwrite(save_path, self.image)
        print(f"Saved annotated image as {save_path}")

    def show_image(self):
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title("Detected Text on PAN Card")
        plt.show()

# --------------------- Example Usage ---------------------
image_path = 'pan card.jpg'
extractor = PANCardEasyOCR(image_path)

ocr_text = extractor.perform_ocr()
print("\nExtracted OCR Text:\n", ocr_text)

pan_data = extractor.extract_fields()
print("\nFinal Extracted Data:\n", json.dumps(pan_data, indent=4))

extractor.draw_boxes()
extractor.show_image()
