from util.inpaint_util   import removeHair

from util import (
    pd,
    cv2,
    os,
    json,
)

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
                label (str): The label of the image
                img_rgb (ndarray): The image in RGB format
                img_gray (ndarray): The image in grayscale format
                mask_fname (str | None): The filename of the mask (if exists)
                original_mask (ndarray | None): The original mask of the image
                blackhat (ndarray): The image after applying blackhat filtering
                thresh (ndarray): The thresholded mask of the image
                img_out (ndarray): The inpainted image
                features (dict): A dictionary to store features extracted from the image

            Methods:
                load(): Loads the image and original mask (if exists) using readImageFile() and removes hair using removeHair()
                set_feature(name, value): Sets a feature in the features dictionary
                get_feature(name): Retrieves a feature from the features dictionary
    """
    def __init__(self, dataset: 'Dataset', filename: str, label: str, mask_fname: str = None) -> None:
        self.dataset = dataset
        self.filename = filename
        self.label = label
        self.img_rgb = None
        self.img_gray = None
        self.mask_fname = mask_fname # Filename from CSV or None
        self.original_mask = None # ndarray or None
        self.blackhat = None
        self.thresh = None
        self.img_out = None
        self.features = {}

    def load(self) -> None:
        # Load the image and possibly its original mask using the parent class's readImageFile method
        self.img_rgb, self.img_gray, self.original_mask = self.dataset.readImageFile(
            file_path = os.path.join(self.dataset.data_dir, self.filename),
            mask_path = os.path.join(self.dataset.data_dir, self.mask_fname) if self.mask_fname else None
        )

        # Apply hair removal
        self.blackhat, self.thresh, self.img_out = removeHair(self.img_rgb, self.img_gray)

    def set_feature(self, name: str, value) -> None:
        # Stores one feature value under self.features[name]
        self.features[name] = value

    def get_feature(self, name: str) -> float | None:
        # Returns the value of the feature stored under self.features[name] or None
        return self.features.get(name)

class Dataset:
    def __init__(self, feature_extractors: dict[str, callable], csv_path: str, data_dir: str, image_col: str = "image_path", mask_col: str = "mask_path", label_col: str = "label") -> None:
        """
            Dataset (parent class):
                Args:
                    feature_extractors (dict): Dictionary containing feature extractor functions (Eg. feature_A, feature_B...)
                    csv_path (str): Absolute path to the CSV file containing metadata
                    data_dir (str): Directory where the images are stored
                    image_col (str): Column name in the CSV that contains image filenames
                    mask_col (str): Column name in the CSV that contains mask filenames (if exists)
                    label_col (str): Column name in the CSV that contains labels
                
                Methods:
                    - readImageFile(file_path): Reads an image from the specified file path and converts it to RGB and grayscale
                    - export_record(rec, out_dir): Saves all information of a Record instance to a specified directory
                    - records_to_dataframe(): Converts each Record instance into a DataFrame for easy access, manipulation and saving of results
        """
        # Save data directory for later read-in purposes
        self.data_dir = data_dir

        # Load metadata and make sure the column containing the filenames exists
        df = pd.read_csv(csv_path)

        # Check if the required columns exist in the DataFrame
        missing = [c for c in (image_col, label_col, mask_col) if c not in df.columns]
        if missing:
            raise KeyError(f"Missing required column(s): {missing}")
        
        # Load each Record
        self.records = []
        for filename, label, mask in zip(df[image_col], df[label_col], df[mask_col]):
            # Create a new Record instance using the filename, label, and mask
            rec = Record(self, filename, label, mask)
            
            # Load the image data and apply hair removal
            rec.load()

            # Extract features using the provided feature extractors
            for feat_name, func in feature_extractors.items():
                # For simplicity, passing the Record instance to each feature extractor function to avoid confusion regarding the input
                rec.set_feature(feat_name, func(rec))
            
            # Append the instance to the records list
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
        original_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) # NEED TO CHECK IF THE ORIGINAL MASK IS VALID (NOT BLANK)

        return img_rgb, img_gray, original_mask

    def export_record(self, rec: Record, out_dir: str) -> dict[str, bool]:
        """
            Saves all information of a Record instance to a specified directory

            Args:
                rec (Record): The Record instance containing the image and its features
                out_dir (str): The directory where the images will be saved
            
            Returns:
                dict[str, bool]: A dictionary containing the status of each image save operation (filename: success)
            
            Prints a warning if the image write operation fails
        """
        # Use the filename only and exclude the extension for the folder name
        name, _ = os.path.splitext(rec.filename)
        folder  = os.path.join(out_dir, name)
        
        # Create the folder if it doesn't exist
        os.makedirs(folder, exist_ok=True)

        # For debugging purposes, save the status of the image saving
        results = {}

        # Create a map of attribute - filename
        to_save = {
            "rgb.png":         rec.img_rgb,
            "gray.png":        rec.img_gray,
            "blackhat.png":    rec.blackhat,
            "thresh.png":      rec.thresh,
            "inpainted.png":   rec.img_out,
        }

        # Only include original_mask if exists
        if rec.original_mask is not None:
            to_save["original_mask.png"] = rec.original_mask
            to_save["original_mask_path.png"] = rec.mask_fname

        # Save each image
        for fname, img in to_save.items():
            # Assemble the full path using the current image filename
            path = os.path.join(folder, fname)

            # If the image is a 3-channel RGB, convert back to BGR for OpenCV
            if img.ndim == 3 and img.shape[2] == 3:
                write_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                write_img = img
            
            # Write the image and save the status
            success = cv2.imwrite(path, write_img)
            
            # Save the status of the write operation
            results[fname] = success

            # Print a warning if the write operation failed
            if not success:
                print(f"[WARN] Failed to write {path}")
        
        # Save the metadata
        meta = {
            "filename": rec.filename,
            "label": rec.label,
            **rec.features, # Unpacks the features dictionary into the metadata
        }

        # Assemble the metadata saving path
        meta_path = os.path.join(folder, "metadata.json")
        try:
            with open(meta_path, "w") as f:
                # obj: data structure to be written to the file
                # f: file object to write to
                # indent: number of spaces to use for indentation in the JSON file (pretty print)
                json.dump(meta, f, indent=2)
            results["metadata.json"] = True
        except Exception as e:
            print(f"[WARN] Could not write metadata.json: {e}")
            results["metadata.json"] = False

        return results

    def records_to__dataframe(self) -> pd.DataFrame:
        """
            Converts each Record instance into a DataFrame for easy access, manipulation and saving of results
            One row corresponds:
                - filename (str)
                - label (str)
                - mask_fname (str | None)
                - has_mask (bool): True if the original mask exists, otherwise False
                - <all features>
            
            Returns:
                pd.DataFrame: A DataFrame containing all the records and their features
        """
        # Create a list of dictionaries to hold the data for each row
        rows = []

        # Iterate through each record and create a dictionary for each one
        for rec in self.records:
            row = {
                "filename":   rec.filename,
                "label":      rec.label,
                "mask_fname": rec.mask_fname,
                "has_mask":   rec.original_mask is not None,
                **rec.features # This syntax unpacks the features dictionary
            }
            # Append the row dictionary to the list of rows
            rows.append(row)
        
        # Convert the list of rows into a DataFrame and return it
        return pd.DataFrame(rows)