from util.img_util import Record
from util import (
    np,
    label,
    regionprops,
    resize,
    KMeans,
    rgb2lab,
    pdist
)

def color_heterogeneity(record: 'Record', n_clusters: int = 4, downscale: float = 1.0, k_blobs: int = 3) -> float:
    """
        The function computes the mean color heterogeneity across the top-k blobs

        Calculation is done by:
            1) Binarizing the lesion mask
            2) Labeling and keeping the k_blobs largest components
            3) For each blob:
                a) Cropping the original image to its bounding box (tightest fit)
                b) Downscaling (optional to increase speed)
                c) Converting to CIE-Lab color space
                d) Clustering into n_clusters using KMeans
                e) Calculating maximum pairwise distance between cluster centroids
            4) Averaging per-blob distances

        Args:
            rec (Record): Record instance containing every bit of information about the image
            n_clusters (int): number of clusters for KMeans
            downscale (float): downscale factor for speedup (default: 1.0, no downscaling)
            k_blobs (int): number of biggest blobs to keep for the heterogeneity calculation

        Returns:
            float: mean color heterogeneity in the interval of [0, ∞), or np.nan if no valid blobs
    """
    # The original mask should be bool type 
    bin_mask = record.image_data["original_mask"].astype(bool)

    # The label function returns a labeled array where each True component has a unique id
    labeled = label(bin_mask)
    
    # The regionprops function scans the labeled array and returns an object (one per blob) that contains stats about it
    props = regionprops(labeled)

    # If no blobs found, return NaN to indicate possible mask issue
    if not props:
        return np.nan

    # Keep the top-k biggest blobs (ensure we don't exceed the number of blobs found)
    props = sorted(props, key=lambda r: r.area, reverse=True)[:min(k_blobs, len(props))]

    # For storing the calculated values
    dists = []

    # Iterate through each blob one-by-one
    for prop in props:
        # Isolating the current blob
        single = (labeled == prop.label)

        # Each region prop contains a bounding box attribute (minr, minc, maxr, maxc) giving the tightest rectangle that contains the blob
        minr, minc, maxr, maxc = prop.bbox
        # Crop the image to the bounding box
        crop_img = record.image_data["original_img"][minr:maxr, minc:maxc]
        print(f"    [INFO] - feature_C.py - minr: {minr}, minc: {minc}, maxr: {maxr}, maxc: {maxc}, crop_img.shape: {crop_img.shape}")

        # Optionally downscale the cropped image and mask to increase speed
        if downscale != 1:
            # .shape attribute's first two elements are height and width, respectively
            h, w = crop_img.shape[:2]

            # Passing the image to the resize function
            # The new scale is computed with the original height and width multiplied by the downscale factor
            # order=0 is to just grab the color/intensity of the single closest original pixel, which is fast but the result is chunky
            # The preserve_range parameter is set to True to keep the original range of pixel values (0-255)
            # The final conversion to uint8 is done to ensure the image is in the correct format for further processing
            crop_img  = resize(crop_img, (int(h*downscale), int(w*downscale)), preserve_range=True).astype(np.uint8)

        # If no pixels are found, skip this blob
        if crop_img.size == 0:
            continue

        # For the heterogeneity metric, we need to know how different two colors look
        # CIE-Lab is a color space designed so that equal Euclidean steps roughly match equal perceived color differences
        # In this case, we have a cropped image ('crop_img') that contains the lesion region
        # The image is in RGB color space, with each pixel represented by a 3-channel (R, G, B) value
        # To compute color heterogeneity, we need to convert these RGB values to the CIE-Lab color space
        # Since rgb2lab expects the input in a 2D array shape (N, 3), where N is the number of pixels and 3 corresponds to the RGB channels:
        #
        #    - crop_img.reshape(-1, 3) reshapes the image array into a 2D array where each row represents one pixel's [R, G, B] values
        #    - The resulting array has shape (N, 3), where N is the total number of pixels in the cropped lesion region
        #
        # After reshaping, the rgb2lab function is applied to convert the RGB values into the Lab color space:
        #    - L: Lightness, ranging from 0 (black) to 100 (white)
        #    - a: Color range from green (-128) to red (+127)
        #    - b: Color range from blue (-128) to yellow (+127)
        #
        # The rgb2lab function converts each RGB value into corresponding Lab values
        # This results in a 2D array with shape (N, 3), where each row represents a pixel's [L, a, b] values
        lab = rgb2lab(crop_img.reshape(-1, 3))
        print(f"    [INFO] - feature_C.py - lab.shape: {lab.shape}")

        # The goal is to pick out the dominant hues from the lesion and quantify how far apart they lie in the Lab space
        # For this, we partition the pixels into "n clusters", which will group the colors into n disjoint subsets
        #       - Have colors in the same group are similar as possible
        #       - Have colors in different groups are as dissimilar as possible
        # Each cluster is represented by its centroid, which is the mean of all the points in that cluster
        # Once having n clusters, the Euclidean distance between any two of them in Lab space is a direct gauge of how far apart the main colros are
        
        # A fall back to avoid potential errors: too few data points, too many clusters, numerical issues...
        try:
            # 1. Randomly pick n initial centroids (cluster centers) from the data points
            # 2. Assign each data point to the cluster with nearest centroid
            # 3. Update the centroid of each cluster by taking the mean of all points assigned to it
            # 4. Repeat steps 2-3 until convergence or a maximum number of iterations is reached
            # 5. Since random initialization can lead to suboptimal minima, the whole algorithm is run n_init times with different seeds
            # random_state = 0 is used to ensure getting the exact same initial centroids and final clustering, making the feature extraction stable
            km = KMeans(n_clusters=n_clusters, n_init=5, random_state=0)
            # Pass the Lab values to the initialized estimator object, which then fits the model
            km.fit(lab)
            centers = km.cluster_centers_ # The final cluster centers (centroids) in (n_clusters, 3) shape
        except Exception:
            continue

        # Next step is to compute the maximum pairwise centroid distance
        #       - The idea is to have an outer loop iterating from 0 to the number of centroids
        #       - An inner loop iterating from i+1 to the number of centroids and computing the distance between the two centroids
        # However to increase efficiency, we use pdist which is implemented in C, which returns a 1D array of pairwise distances
        # Then just take the maximum and append it to the list of distances
        dists.append(pdist(centers).max())

    # Finally, return the mean of the top-k distances (or np.nan as a fallback in case of no valid blobs)
    return float(np.nanmean(dists)) if dists else np.nan