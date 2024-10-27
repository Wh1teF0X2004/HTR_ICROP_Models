import cv2
import numpy as np
import os


def crop():
    # Load the template image (reference form)
    template_image_path = r"C:\Users\Owner\Documents\FYP\automated-health-information-system\MDS2_AHIS\HTR_Model\SimpleHTR-master\src\Form_Template.jpg"
    template_image = cv2.imread(template_image_path, cv2.IMREAD_GRAYSCALE)
    if template_image is None:
        raise ValueError(f"Template image not found at {template_image_path}")

    # Load the filled form image (scanned form from patient)
    filled_image_path = r"C:\Users\Owner\Documents\FYP\automated-health-information-system\MDS2_AHIS\data_files\patientRegistrationForm.jpg"
    filled_image = cv2.imread(filled_image_path, cv2.IMREAD_GRAYSCALE)
    if filled_image is None:
        raise ValueError(f"Filled image not found at {filled_image_path}")

    # Resize filled image to match template dimensions
    template_height, template_width = template_image.shape
    filled_image = cv2.resize(filled_image, (template_width, template_height))

    # Detect keypoints and descriptors using ORB detector
    orb = cv2.ORB_create()
    keypoints_template, descriptors_template = orb.detectAndCompute(template_image, None)
    keypoints_filled, descriptors_filled = orb.detectAndCompute(filled_image, None)

    # Match descriptors using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors_template, descriptors_filled)

    # Sort matches based on distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Use homography to align the filled image with the template
    if len(matches) >= 4:
        src_pts = np.float32([keypoints_template[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        dst_pts = np.float32([keypoints_filled[m.trainIdx].pt for m in matches]).reshape(-1, 2)

        # Compute the homography matrix
        matrix, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        aligned_image = cv2.warpPerspective(filled_image, matrix, (template_width, template_height))

        # Define the field coordinates for cropping
        fields_coordinates_adjusted = {
            'FirstName': (820, 546, 672, 76),
            'LastName': (820, 630, 670, 74),
            'BirthDate': (822, 710, 666, 74),
            'Gender': (820, 790, 666, 74),
            'Nationality': (822, 872, 662, 72),
            'IC_Passport': (822, 952, 660, 70),
            'MaritalStatus': (824, 1034, 656, 114),
            'PhoneNo': (820, 1158, 662, 70),
            'Email': (820, 1238, 660, 68),
            'Address': (822, 1318, 658, 68),
            'City': (820, 1396, 660, 70),
            'State': (822, 1476, 660, 70),
            'ZipCode': (822, 1560, 654, 66),
            'EC_FirstName': (822, 1722, 654, 66),
            'EC_LastName': (822, 1800, 654, 69),
            'Relationship': (822, 1887, 654, 60),
            'EC_ContactNo': (819, 1962, 654, 69)
        }

        # Extract the filled image filename (without extension) to use as directory name
        filled_image_filename = os.path.splitext(os.path.basename(filled_image_path))[0]
        # save_directory = os.path.join(
        #     r'C:\Users\Alicia\OneDrive - Monash University\Desktop\FIT3164\Project\automated-health-information-system\MDS2_AHIS\HTR_Model\SimpleHTR-master\data',
        #     filled_image_filename)
        save_directory = r'C:\Users\Owner\Documents\FYP\automated-health-information-system\MDS2_AHIS\data_files\crop_images'

 
        # Create directory if it does not exist
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Log file to store cropped field paths
        log_file_path = os.path.join(save_directory, 'cropped_fields_log.txt')
        with open(log_file_path, 'w') as log_file:
            # Loop through the coordinates and crop each field
            for field_name, (x, y, w, h) in fields_coordinates_adjusted.items():
                cropped_field = aligned_image[y:y + h, x:x + w]
                cropped_image_path = os.path.join(save_directory, f'{field_name}.jpg')
                cv2.imwrite(cropped_image_path, cropped_field, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

                # Log the path of the saved image
                log_file.write(f'{field_name}: {cropped_image_path}\n')

        #print(f"Cropping completed successfully. Images saved in directory: {save_directory}")
        print("Cropping completed successfully.")
    else:
        print("Not enough matches found to align images.")


# Run the crop function
if __name__ == "__main__":
    crop()
