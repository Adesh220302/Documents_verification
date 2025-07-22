import easyocr
import cv2
import re
import matplotlib.pyplot as plt

class AadhaarCardExtractor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        self.reader = easyocr.Reader(['en'])
        self.results = self.reader.readtext(image_path)
        self.data = {
            "Name": None,
            "DOB": None,
            "Gender": None,
            "Aadhaar Number": None
        }

    def extract_fields(self):
        for bbox, text, conf in self.results:
            text = text.strip()

            # Draw bounding box
            top_left = tuple(map(int, bbox[0]))
            bottom_right = tuple(map(int, bbox[2]))
            cv2.rectangle(self.image, top_left, bottom_right, (0, 255, 0), 1)
            cv2.putText(self.image, text, (top_left[0], top_left[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

            # Extract fields
            if re.search(r'\d{2}/\d{2}/\d{4}', text):
                self.data["DOB"] = text
            elif re.match(r'^\d{4} \d{4} \d{4}$', text):
                self.data["Aadhaar Number"] = text
            elif text.lower() in ['male', 'female', 'other']:
                self.data["Gender"] = text
            elif self.data["Name"] is None and "DOB" not in text:
                self.data["Name"] = text

        return self.data

    def show_image(self):
        image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(6,4))
        plt.imshow(image_rgb)
        plt.axis('off')
        plt.title("Aadhaar Card OCR with Bounding Boxes")
        plt.show()

# --------------------------- Example Usage --------------------------- #
image_path = "adhar_card.jpeg"
extractor = AadhaarCardExtractor(image_path)
aadhaar_data = extractor.extract_fields()

print("Extracted Aadhaar Data:")
print(aadhaar_data)

extractor.show_image()
