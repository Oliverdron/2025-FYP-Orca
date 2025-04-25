from util import (
    pd, cv2, os,
    removeHair
) 

class Dataset:
    def __init__(self, feature_extractors: dict, csv_path: str, data_dir: str, filename_col: str = "image_path", label_col: str = "label"):
        """
            Dataset (parent class):
                Args:
                    feature_extractors (dict): Dictionary containing feature extractor functions (Eg. feature_A, feature_B...)
                    csv_path (str): Absolute path to the CSV file containing metadata
                    data_dir (str): Directory where the images are stored
                    filename_col (str): Column name in the CSV that contains image filenames
                    label_col (str): Column name in the CSV that contains labels
                    records (list): List of Record instances, each representing an image and its features
                
                Methods:
                    - readImageFile(file_path): Reads an image from the specified file path and converts it to RGB and grayscale
                    - saveImageFile(img_rgb, file_path): Saves the provided image in RGB format to the specified file path
                    - records_to__dataframe(): Saves the features of each image in a DataFrame for easy access and manipulation or saving results
        """
        # Save data directory for later read-in purposes
        self.data_dir = data_dir

        # Load metadata and make sure the column containing the filenames exists
        df = pd.read_csv(csv_path)
        if filename_col not in df.columns or label_col not in df.columns:
            raise KeyError(f"Missing one of '{filename_col}' or '{label_col}' in {csv_path}")
        
        # Load each Record
        self.records = []
        for filename, label in zip(df[filename_col], df[label_col]):
            # Create a new Record instance for each filename and label
            rec = self.Record(filename, label)
            
            # Load the image data and apply hair removal
            rec.load()

            # Extract features using the provided feature extractors
            for feat_name, func in feature_extractors.items():
                # !!!! CHECK IF ALL EXTRACTION FEATURES USE THE SAME IMAGE !!!!
                rec.set_feature(feat_name, func(rec.img_out))
            
            # Append the instance to the records list
            self.records.append(rec)

    def readImageFile(self, file_path: str):
        """
            Reads an image from the specified file path and converts it to RGB and grayscale

            Args:
                file_path (str): The path to the image file to be read

            Returns:
                tuple: A tuple containing:
                    - img_rgb (ndarray): The image in RGB format
                    - img_gray (ndarray): The image in grayscale format
                
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

        return img_rgb, img_gray

    def saveImageFile(self, img_rgb, file_path: str) -> bool:
        """
            Saves the provided image in RGB format to the specified file path

            Args:
                img_rgb (ndarray): The image in RGB format to be saved
                file_path (str): The path where the image will be saved
                
            Returns:
                bool: True if the image was saved successfully, false otherwise
                
            Raises:
                Exception: If an error occurs during the saving process
        """
        try:
            # The first argument to cv2.imwrite() is the file path (file_path), where the image should be saved
            # The second argument is the image, where the it's being converted from RGB to BGR color format because OpenCV uses BGR by default
            # It also returns a True of False value to verify that the image was saved properly
            success = cv2.imwrite(file_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            if not success:
                print(f"Failed to save the image to {file_path}")
            return success

        except Exception as e:
            print(f"Error saving the image: {e}")
            return False

    def records_to__dataframe(self) -> pd.DataFrame:
        """
            Saves the features of each image in a DataFrame for easy access and manipulation or saving results
        """
        rows = []
        for rec in self.records:
            # The **rec.features syntax unpacks that dictionary into the new dict literal
            rows.append({"filename": rec.filename, "label:": rec.label, **rec.features})
        return pd.DataFrame(rows)
    
    class Record:
        """
            Record (child class):
                Attributes:
                    filename (str): The name of the image file
                    label (str): The label of the image
                    img_rgb (ndarray): The image in RGB format
                    img_gray (ndarray): The image in grayscale format
                    blackhat (ndarray): The image after applying blackhat filtering
                    thresh (ndarray): The thresholded mask of the image
                    img_out (ndarray): The inpainted image
                    features (dict): A dictionary to store features extracted from the image
                    
                Methods:
                    load(): Loads the image using readImageFile() and removes hair using removeHair()
                    set_feature(name, value): Sets a feature in the features dictionary
                    get_feature(name): Retrieves a feature from the features dictionary
        """
        def __init__(self, filename: str, label: str):
            self.filename = filename
            self.label = label
            self.img_rgb = None
            self.img_gray = None
            self.blackhat = None
            self.thresh = None
            self.img_out = None
            self.features = {}

        def load(self):
            # Load the image using the parent class's readImageFile method
            full_path = os.path.join(self._dataset.data_dir, self.filename)
            self.img_rgb, self.img_gray = self._dataset.readImageFile(full_path)
            # Apply hair removal
            self.blackhat, self.thresh, self.img_out = removeHair(self.img_rgb, self.img_gray) # COULD BE IMPROVED BY PERSONALIZED PARAMETERS?

        def set_feature(self, name: str, value):
            # Stores one feature value under self.features[name]
            self.features[name] = value

        def get_feature(self, name: str):
            # Returns the value of the feature stored under self.features[name]
            return self.features.get(name)