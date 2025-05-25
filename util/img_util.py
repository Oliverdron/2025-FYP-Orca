from util.img_preprocess_util import PreProcessor
from util import (
    pd,
    cv2,
    os,
    sys,
    time,
)

CANCEROUS = ("BCC","MEL","SCC")
METADATACSV = "metadata.csv"

class Record:
    """
        Record (child class):
            Param:
                dataset (Dataset): The parent Dataset instance
                id (str): The ID of a patient as it is possible to have multiple images from one patient
                image_fname (str): The name of the image file
                label (str): The label of the image
                mask_fname (str | None): The filename of the mask (if exists)

            Methods:
                load(): Loads the image and original mask (if exists) using readImageFile() and removes hair using removeHair()
                set_feature(name, value): Sets a feature in the features dictionary
                get_feature(name): Retrieves a feature from the features dictionary
    """
    def __init__(self, dataset: 'Dataset', id: str, image_fname: str, mask_fname: str, label: str) -> None:
        self.dataset = dataset # Stores the parent Dataset instance
        self.meta_data = {
            "patient_id": id, # Stores the patient's id as recurrence is probable
            "image_fname": image_fname.split('.')[0], # The name of the image file in the CSV file (without the extension)
            "extension": image_fname.split('.')[1], # The extension of the file
            "label_binary": True if label in CANCEROUS else False, # The binary label of the image: True if cancerous; False if not
            "label_category": label, # The categorical label of the image, one of ["BCC","ACK""MEL","SEK","NEV","SEC"]
            "mask_fname": mask_fname, # Filename from CSV or None
        }

        self.image_data = {
            "original_img": None,  # Original RGB image
            "original_mask": None, # Original (provided) mask or None
            "grayscale_img": None,  # Grayscale version of the original image
            "denoised_img": None,  # Denoised image (after bilateral or NLMD denoising)
            "blackhat_img": None,  # Result of Black-Hat morphological operation
            "threshold_hair_mask": None,  # Binary mask for inpainting
            "inpainted_img": None,  # Image after hair removal
            "enhanced_img": None,  # Image after adaptive histogram equalization
            "threshold_segm_mask": None # Binary mask for lesion segmentation
        }

        self.features = {
            "hair_label": None # Indicates how much hair there is on the original image
        }

    def load(self) -> None:
        # Load the image and possibly its original mask using the parent class's readImageFile method
        self.image_data["original_img"], self.image_data["grayscale_img"], self.image_data["original_mask"] = self.dataset.readImageFile(
            file_path = os.path.join(self.dataset.data_dir, "images", f"{self.meta_data['image_fname']}.{self.meta_data['extension']}"),
            mask_path = os.path.join(self.dataset.data_dir, "lesion_masks", f"{self.meta_data['mask_fname']}") if self.meta_data["mask_fname"] else None
        )

    def set_feature(self, name: str, value) -> None:
        # Stores one feature value under self.features[name]
        print(f"    [INFO] - img_util.py - Setting feature {name!r} to {value}")
        self.features[name] = value

    def get_feature(self, name: str) -> float | None:
        # Returns the value of the feature stored under self.features[name] or None
        return self.features.get(name)

class Dataset:
    def __init__(
            self, feature_extractors: dict[str, callable], base_dir: str, columns: list = ["patient_id", "image_path", "mask_path", "label"], shuffle: bool = False, limit: int = None) -> None:
        """
            Dataset (parent class):
                Param:
                    feature_extractors (dict): Dictionary containing feature extractor functions (Eg. feature_A, feature_B...)
                    base_dir (str): The base directory path
                    columns (list): A list that contains the necessary column names in the CSV file (Eg. patient's id, image file name, mask file name, label)
                    shuffle (bool): Whether to shuffle the records (default: False)
                    limit (int): Limit the number of records processed (default: None, which means no limit)
                
                Methods:
                    - readImageFile(file_path): Reads an image from the specified file path and converts it to RGB and grayscale
                    - export_record(rec, out_dir): Saves all information of a Record instance to a specified directory
                    - records_to_dataframe(): Converts each Record instance into a DataFrame for easy access, manipulation and saving of results
        """
        # Save base directory for later read-in purposes
        self.base_dir = base_dir
        self.data_dir = base_dir / "data"
    	
        # Check if the metadata exists
        csv_path = base_dir / METADATACSV
        if not csv_path.exists():
            sys.exit(f"[ERROR] - img_util.py - metadata.csv not found in {base_dir}")
        
        # Load metadata 
        df = pd.read_csv(csv_path)

        # Check if the required columns exist in the DataFrame
        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise KeyError(f"[ERROR] - img_util.py - Missing required column(s): {missing}")
        
        # Shuffle the dataset if needed
        if shuffle:
            df = df.sample(frac=1).reset_index(drop=True)

        # Load each Record
        self.records = []
        for id, image_fname, mask_fname, label in zip(*[df[col] for col in columns]):
            # Create a new Record instance using the filename, mask and label
            t0 = time.perf_counter()
            rec = Record(self, id, image_fname, mask_fname, label)
            t1 = time.perf_counter()
            elapsed = t1 - t0
            print(f"[INFO] - img_util.py - Record {rec.meta_data["image_fname"]!r} took {elapsed:.4f}s to initialize")
            
            # Load the image data and apply hair removal
            t0 = time.perf_counter()
            rec.load()
            t1 = time.perf_counter()
            elapsed = t1 - t0
            print(f"    [INFO] - img_util.py - Record {rec.meta_data["image_fname"]!r} took {elapsed:.4f}s to load")
            
            # Before any extraction, apply image pre-processing method
            # It won't return anything, rather it'll use the rec.image_data dictionary to save different phases of pre-processing/modified images
            PreProcessor(rec)

            # Extract features using the provided feature extractors
            for feat_name, func in feature_extractors.items():
                # For simplicity, passing the Record instance to each feature extractor function to avoid confusion regarding the input
                t0 = time.perf_counter()
                value = func(rec)
                t1 = time.perf_counter()
                elapsed = t1 - t0
                print(f"    [INFO] - img_util.py - Feature {feat_name!r} took {elapsed:.4f}s, value={value}")
                rec.set_feature(f"{feat_name}", value)
            
            # Append the Record instance to the records list
            self.records.append(rec)
            print(f"[INFO] - img_util.py - Current number of images: {len(self.records)}")
            if limit:
                if len(self.records) >= limit:
                    print(f"[INFO] - img_util.py - Reached the limit of {limit} records for testing.")
                    break
                # Since we have a limit, it is a test-case, so save all information of a Record (including images)
                self.export_record(rec, "dataset.csv", True)
            else:
                # Call the export_record method to save the Record instance's data to a CSV file
                self.export_record(rec, "dataset.csv")

    def readImageFile(self, file_path: str, mask_path: str = None) -> tuple:
        """
            Reads an image from the specified file path and converts it to RGB and grayscale

            Args:
                file_path (str): The path of the image file to be read
                mask_path (str): The path of the mask file to be read (if exists)

            Returns:
                tuple:
                    - img_rgb (ndarray): The image in RGB format
                    - img_gray (ndarray): The image in grayscale format
                    - original_mask (ndarray | None): The original mask of the image (if exists)
                
            Raises:
                ValueError: If the image cannot be loaded from the file path
        """
        # The function reads the image and returns it as a NumPy array in BGR
        img_bgr = cv2.imread(file_path)

        # If the file doesn't exist/the file is corrupted/the file is not a valid image, it will return None
        if img_bgr is None:
            raise ValueError(f"Unable to load image at {file_path}")

        # The function returns the image in RGB format
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # The function returns the grayscale version of the image
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        # If a mask path is provided, read the mask image in grayscale format
        # The read function returns None if the file does not exist
        # But first check if the mask_path is a string (not None)
        original_mask = None # Initialize original_mask to None to avoid uninitialized variable error while returning
        if mask_path:
            original_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        print(f"    [INFO] - img_util.py - Loaded image {file_path} with shape {img_rgb.shape} | mask {mask_path} with shape {original_mask.shape if original_mask is not None else None}")

        return img_rgb, img_gray, original_mask

    def export_record(self, rec: Record, csv_name: str, save_images: bool = False) -> None:
        """
            Saves all text information of a Record instance to the specified CSV file

            Args:
                rec (Record): The Record instance containing the image and its features
                csv_name (str): The csv file's name, which we use to build the absolute path
            
            Prints a warning if the writing operation fails
        """
        # Create a map of attributes to be saved
        to_save = {
            **rec.meta_data,    # Unpacks the metadata dictionary
            **rec.features      # Unpacks the features dictionary
        }

        try:
            csv_path = self.base_dir / csv_name
            # Decide whether to write the header: only if the file doesn’t yet exist or is empty
            write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
            # Convert the dictionary to a DataFrame and save it to the CSV file
            pd.DataFrame([to_save]).to_csv(csv_path, mode='a', header=write_header, index=False)
            print(f"    [INFO] - img_util.py - Appended {rec.meta_data["image_fname"]} to {csv_path}")

            # Save the images if save_images is True
            if save_images:
                # Construct the folder path based on the filename (without extension)
                file_name = rec.meta_data["image_fname"]
                result_dir = self.base_dir / "result" / "testing" / file_name  # Create the result folder path

                # Create the directory if it does not exist (overwrite if needed)
                result_dir.mkdir(parents=True, exist_ok=True)

                # Iterate through the image_data dictionary and save images using the keys as filenames
                for key, image in rec.image_data.items():
                    if image is not None:  # Ensure image is not None before saving
                        # Construct the full filename for each image
                        output_filename = f"{file_name}_{key}.png"
                        # Save the image (convert to BGR if necessary for OpenCV)
                        if len(image.shape) == 3 and image.shape[2] == 3:  # RGB image
                            cv2.imwrite(str(result_dir / output_filename), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                        else:  # Grayscale or other formats
                            cv2.imwrite(str(result_dir / output_filename), image)
                        print(f"    [INFO] - img_util.py - Saved {output_filename} in {result_dir}")

        # Raise a warning if the writing operation fails
        except Exception as e:
            print(f"[WARNING] - img_util.py - Could not append to {csv_path}: {e}")