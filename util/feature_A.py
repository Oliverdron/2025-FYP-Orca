from util.img_util import Record

from util import (
    np,
    rotate,
    label,
    regionprops
)

def asymmetry(record: 'Record', k_blobs: int = 3, n_angles: int = 16) -> float:
    """
        Compute the mean rotation-invariant asymmetry across the top-k blobs in the Record's mask.

        Compute asymmetry score by:
            1) Binarizing the lesion mask
            2) Labeling and keeping the k_blobs largest components
            3) For each blob:
                a) Rotating via nearest-neighbor through n_angles between 0-180 degrees
                b) Tight-cropping to its bounding box
                c) Splitting in half horizontally/vertically and XOR'ing
                d) Normalizing mismatch by blob area
            4) Averaging per-blob asymmetry scores

        Args:
            rec (Record): Record instance containing every bit of information about the image
            k_blobs (int): number of biggest blobs to keep for the asymmetry calculation
            n_angles (int): number of sample rotations between 0 degrees and 180 degrees
        
        Returns:
            float: mean asymmetry in [0, 1] across the top-k blobs, or np.nan if no blobs are found
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

    # Storing the blobs' scores, then taking its mean later
    blob_scores = []

    # Precompute evenly spaced angles
    angles = np.linspace(0, 180, n_angles, endpoint=False)

    # Iterate through each blob one-by-one
    for prop in props:
        # Isolating the current blob
        single = (labeled == prop.label)

        # To store each rotation's asymmetry score
        scores = []

        # Iterate through each angle
        for angle in angles:
            # Rotate with nearest-neighbor -> stays boolean
            rot = rotate(single.astype(float), # Transformation works on numeric arrays
                        angle,
                        # When rotating an image, most of the new pixel grid does not line up exactly with the old one
                        # order=0 is to just grab the color/intensity of the single closest original pixel, which is fast but the result is chunky
                        order=0,
                        preserve_range=True # Avoid rescaling the data
                        ).astype(bool) # Turns the numeric array back into a boolean one

            rows = np.any(rot, axis=1) # Find rows with at least one True value
            cols = np.any(rot, axis=0) # Find columns with at least one True value

            # np.where returns the indices of True rows/columns
            # np.where(rows/cols)[0] extracts the array of indices
            row_idx = np.where(rows)[0]
            col_idx = np.where(cols)[0]
            
            # If no rows or columns are found, skip this rotation
            if row_idx.size == 0 or col_idx.size == 0:
                continue

            # np.where(rows)[0][[0, -1]] gives the first (topmost\leftmost) and last (bottommost/rightmost) indices of the True values
            rmin, rmax = row_idx[[0, -1]]
            cmin, cmax = col_idx[[0, -1]]

            crop = rot[rmin:rmax+1, cmin:cmax+1] # Slice out the skin lesion area (+1 to include the last index)

            # Take the center of the blob to split it into 4 quadrants later
            cy2, cx2 = crop.shape[0] / 2, crop.shape[1] / 2

            top = crop[: int(np.ceil(cy2)), :] # Top half is from the first row to the center row (inclusive)
            bottom = crop[int(np.floor(cy2)) :, :] # Bottom half is from the center row (exclusive) to the last row
            left = crop[:, : int(np.ceil(cx2))] # Left half is from the first column to the center column (inclusive)
            right = crop[:, int(np.floor(cx2)) :] # Right half is from the center column (exclusive) to the last column

            # 5) XOR-based mismatch with its flipped counterpart to find the asymmetry
            h_xor = np.logical_xor(top, np.flip(bottom, 0))
            v_xor = np.logical_xor(left, np.flip(right, 1))

            # The area of the skins lesion is the sum of all pixels in the crop
            area = crop.sum()
            # Should not happen, but if the area is 0, the score is 0
            if area == 0:
                scores.append(0.0)
            else:
                # The score is the sum of the XORs divided by the area (normalized to have a value between 0 and 1)
                scores.append((h_xor.sum() + v_xor.sum()) / (2 * area))
        
        # Append the mean of the scores for this blob to the list
        blob_scores.append(float(np.mean(scores)))

    # Overall asymmetry is the average across all sampled rotations
    return float(np.mean(blob_scores))