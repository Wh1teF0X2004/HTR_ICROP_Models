## Third Attempt
import cv2
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

class ImageCropper:
    def __init__(self, input_image_path: str, output_dir: str):
        """Initializes the image cropper with the input image and output directory."""
        self.input_image_path = input_image_path
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def crop_fields(self, fields: dict) -> None:
        """Crops regions from the image based on the provided fields dictionary."""
        img = cv2.imread(self.input_image_path)
        if img is None:
            logging.error(f"Error: Image not found at {self.input_image_path}")
            return

        for field, coords in fields.items():
            x, y, w, h = coords
            cropped_img = img[y:y + h, x:x + w]
            output_path = os.path.join(self.output_dir, f"{field}.png")
            try:
                cv2.imwrite(output_path, cropped_img)
                logging.info(f"Saved {field} field to {output_path}")
            except Exception as e:
                logging.error(f"Failed to save {field}: {e}")

    @staticmethod
    def get_field_coordinates() -> dict:
        """Returns predefined coordinates for the fields to be cropped."""
        return {
            'FirstName': (620, 456, 470, 54),
            'LastName': (620, 508, 470, 54),
            'BirthDate': (620, 558, 470, 58),
            'Gender': (620, 610, 472, 56),
            'Nationality': (618, 664, 472, 54),
            'IC_Passport': (618, 714, 472, 58),
            'MaritalStatus': (620, 770, 472, 80),
            'PhoneNo': (620, 850, 472, 54),
            'Email': (620, 900, 472, 58),
            'Address': (620, 954, 472, 56),
            'City': (620, 1006, 472, 54),
            'State': (618, 1060, 472, 52),
            'ZipCode': (620, 1110, 472, 54),
            'EC_FirstName': (618, 1218, 474, 52),
            'EC_LastName': (618, 1268, 472, 56),
            'Relationship': (620, 1320, 472, 56),
            'EC_ContactNo': (618, 1374, 472, 56)
        }

def main():
    input_image_path = r'C:\Users\Eunice Lee\OneDrive - Monash University\Desktop\HTR Model\SimpleHTR-master\Image_Cropping_Model\FYP_ImageProcessing\PRF_DarkBlue.JPG'
    output_dir = r'C:\Users\Eunice Lee\OneDrive - Monash University\Desktop\HTR Model\SimpleHTR-master\data\output_fields'

    cropper = ImageCropper(input_image_path, output_dir)
    field_coordinates = cropper.get_field_coordinates()
    cropper.crop_fields(field_coordinates)

if __name__ == "__main__":
    main()
