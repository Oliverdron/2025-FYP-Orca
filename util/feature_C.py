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
                a) Cropping to its bounding box (tightest fit)
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
            float: mean color heterogeneity in the interval of [0, infinity), or np.nan if no valid blobs
    """
    # Thresh image contains either 0 or 255 values, so we need to convert it to boolean for easier processing
    bin_mask = record.thresh.astype(bool)

    # The label function returns a labeled array where each True component has a unique id
    labeled = label(bin_mask.astype(int))
    
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
        # Crop the image and mask to the bounding box
        crop_img = record.img_out[minr:maxr, minc:maxc]
        crop_mask = single[minr:maxr, minc:maxc]

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
            crop_mask = resize(crop_mask.astype(float), (int(h*downscale), int(w*downscale)), order=0, preserve_range=True).astype(bool)

        # Extract the true RGB pixels from the cropped image using the mask
        pix = crop_img[crop_mask]

        # If no pixels are found, skip this blob
        if pix.size == 0:
            continue

        # For the heterogeneity metric we need to know how different two colors look
        # CIE-Lab is a color space designed so that equal Euclidean steps roughly match equal perceived color differences
        # Now, since pix is an array shape (N, 3) where each row is one pixel's [R, G, B] values, we need to reshape it so the rgb2lab can work with it
        # pix.reshape(-1, 1, 3):
        #       - -1: means automatically assign the first dimension, which corresponds to the number of rows in the image
        #       - 1: to keep the second dimension as is, which corresponds to the number of columns in the image
        #       - 3: to keep the third dimension as is, which corresponds to the number of color channels (RGB) for each entry
        # Which is fed into the rgb2lab function (that expects a 3D array)
        # Then, reshape it back to a 2D array with shape (N, 3) where each row is one pixel's [L, a, b] values
        #       - L: is lightness (0 = black to 100 = white)
        #       - a: runs from green to red (-128 to 127)
        #       - b: runs from blue to yellow (-128 to 127)
        # It is important to do so, as the KMeans expects a 2D array
        lab = rgb2lab(pix.reshape(-1,1,3)).reshape(-1,3)

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
            km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
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