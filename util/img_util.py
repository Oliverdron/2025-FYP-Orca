from util.inpaint_util_demo import removeHair
from util import (
    pd,
    cv2,
    os,
    sys,
    time,
)

CANCEROUS = ("BCC","MEL","SCC")

class Record:
    """
        Record (child class):
            Args:
                dataset (Dataset): The parent Dataset instance
                filename (str): The name of the image file
                label (str): The label of the image
                mask_fname (str | None): The filename of the mask (if exists)

            Attributes:
                dataset (Dataset): The parent Dataset instance
                filename (str): The name of the image file
                label_categorical (str): The categorical label of the image, one of ["BCC","ACK""MEL","SEK","NEV","SEC"]
                label_binary (boolean): The binary label of the image, True if cancerous, False if not
                img_rgb (ndarray): The image in RGB format
                img_gray (ndarray): The image in grayscale format
                mask_fname (str | None): The filename of the mask (if exists)
                original_mask (ndarray | None): The original mask of the image
                blackhat (ndarray): The image after applying blackhat filtering
                thresh (ndarray): The thresholded mask of the image
                img_out (ndarray): The inpainted image
                img_hair_label (int): The label indicating the amount of hair in the image
                features (dict): A dictionary to store features extracted from the image

            Methods:
                load(): Loads the image and original mask (if exists) using readImageFile() and removes hair using removeHair()
                set_feature(name, value): Sets a feature in the features dictionary
                get_feature(name): Retrieves a feature from the features dictionary
    """
    def __init__(self, dataset: 'Dataset', filename: str, label: str, mask_fname: str = None) -> None:
        self.dataset = dataset
        self.filename = filename
        self.label_categorical = label
        self.label_binary = True if label in CANCEROUS else False 
        self.img_rgb = None
        self.img_gray = None
        self.mask_fname = mask_fname # Filename from CSV or None
        self.original_mask = None # ndarray or None
        self.blackhat = None
        self.thresh = None
        self.img_out = None
        self.img_hair_label = None
        self.features = {}

    def load(self) -> None:
        # Load the image and possibly its original mask using the parent class's readImageFile method
        self.img_rgb, self.img_gray, self.original_mask = self.dataset.readImageFile(
            file_path = os.path.join(self.dataset.data_dir, "images", self.filename),
            mask_path = os.path.join(self.dataset.data_dir, "lesion_masks", self.mask_fname) if isinstance(self.mask_fname, str) else None
        )

        # Apply hair removal
        self.blackhat, self.thresh, self.img_out, self.img_hair_label = removeHair(self.img_rgb, self.img_gray)

    def set_feature(self, name: str, value) -> None:
        # Stores one feature value under self.features[name]
        self.features[name] = value

    def get_feature(self, name: str) -> float | None:
        # Returns the value of the feature stored under self.features[name] or None
        return self.features.get(name)
        

class Dataset:
    def __init__(self, feature_extractors: dict[str, callable], base_dir: str, image_col: str = "image_path", mask_col: str = "image_mask_path", label_col: str = "label") -> None:
        """
            Dataset (parent class):
                Args:
                    feature_extractors (dict): Dictionary containing feature extractor functions (Eg. feature_A, feature_B...)
                    base_dir (str): The base directory path
                    image_col (str): Column name in the CSV that contains image filenames
                    mask_col (str): Column name in the CSV that contains mask filenames (if exists)
                    label_col (str): Column name in the CSV that contains labels
                
                Methods:
                    - readImageFile(file_path): Reads an image from the specified file path and converts it to RGB and grayscale
                    - export_record(rec, out_dir): Saves all information of a Record instance to a specified directory
                    - records_to_dataframe(): Converts each Record instance into a DataFrame for easy access, manipulation and saving of results
        """
        # Save base directory for later read-in purposes
        self.base_dir = base_dir
        self.data_dir = base_dir / "data"

        csv_path = base_dir / "metadata.csv"
        if not csv_path.exists():
            sys.exit(f"[ERROR] - img_util.py - LINE 97 - metadata.csv not found in {base_dir}")
        
        # Load metadata 
        df = pd.read_csv(csv_path)

        # Check if the required columns exist in the DataFrame
        missing = [c for c in (image_col, label_col, mask_col) if c not in df.columns]
        if missing:
            raise KeyError(f"[ERROR] - img_util.py - LINE 105 - Missing required column(s): {missing}")
        
        # Load each Record
        self.records = []
        for filename, label, mask in zip(df[image_col], df[label_col], df[mask_col]):
            print(f"[INFO] - img_util.py - LINE 110 - Current number of images: {len(self.records)}")

            # Create a new Record instance using the filename, label, and mask
            t0 = time.perf_counter()
            rec = Record(self, filename, label, mask)
            t1 = time.perf_counter()
            elapsed = t1 - t0
            print(f"[INFO] - img_util.py - LINE 117 - Record {filename!r} took {elapsed:.4f}s to initialize")
            
            # Load the image data and apply hair removal
            t0 = time.perf_counter()
            rec.load()
            t1 = time.perf_counter()
            elapsed = t1 - t0
            print(f"[INFO] - img_util.py - LINE 124 - Record {filename!r} took {elapsed:.4f}s to load")

            # Extract features using the provided feature extractors
            for feat_name, func in feature_extractors.items():
                # For simplicity, passing the Record instance to each feature extractor function to avoid confusion regarding the input
                t0 = time.perf_counter()
                value = func(rec)
                t1 = time.perf_counter()
                elapsed = t1 - t0
                # Scale the feature values using StandardScaler
                # Still have to test if this works as intended, but feature values should be standardized
                #scaler = StandardScaler()
                #value = scaler.fit_transform(value.reshape(-1, 1)).flatten()
                rec.set_feature(f"{feat_name}", value)
                print(f"[INFO] - img_util.py - LINE 138 - Feature {feat_name!r} took {elapsed:.4f}s, value={value}")
            
            # Call the export_record method to save the Record instance's data to a CSV file
            self.export_record(rec, os.path.join(self.base_dir, "dataset.csv"))

            # Append the Record instance to the records list
            self.records.append(rec)

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

        print(f"[INFO] - img_util.py - LINE 183 - Loaded image {file_path} with shape {img_rgb.shape} | mask {mask_path} with shape {original_mask.shape if original_mask is not None else None}")

        return img_rgb, img_gray, original_mask

    def export_record(self, rec: Record, csv_path: str):
        """
            Saves all text information of a Record instance to the specified CSV file

            Args:
                rec (Record): The Record instance containing the image and its features
                csv_path (str): The csv file's absolute path, where the information will be saved
            
            Prints a warning if the writing operation fails
        """
        # Create a map of attributes to be saved
        to_save = {
            "filename":             rec.filename,
            "label_binary":         rec.label_binary,
            "label_categorical":    rec.label_categorical  
            **rec.features      # Unpacks the features dictionary into the metadata
        }

        try:
            # Decide whether to write the header: only if the file doesn’t yet exist or is empty
            write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
            # Convert the dictionary to a DataFrame and save it to the CSV file
            pd.DataFrame([to_save]).to_csv(csv_path, mode='a', header=write_header, index=False)
            print(f"[INFO] - img_util.py - LINE 210 - Appended {rec.filename} to {csv_path}")
        # Raise a warning if the writing operation fails
        except Exception as e:
            print(f"[WARNING] - img_util.py - LINE 213 - Could not append to {csv_path}: {e}")