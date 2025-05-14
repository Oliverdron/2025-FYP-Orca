from util.img_util import Record

from util import (
    np,
    ConvexHull,
    label,
    regionprops
)

def border_irregularity(record: 'Record', k_blobs: int = 3) -> float | np.nan:
    """
        The function computes the mean combined index M = I * (1 - C) over the top-k blobs using the following steps:
            1) I = perimeter^2 / (4 * pi * area) -> spikiness measure
                A perfect circle has I = 1, while a more irregular shape has I > 1
            2) C = area / area_hull -> convexity measure
                A perfect convex shape has C = 1, while a more irregular shape has C < 1
            3) M = I * (1 - C) -> combined index
                So when a blob is both spiky (I > 1) and highly concave (1 - C appr = 1) M becomes large
                If it is either smooth (I = 1) or convex (C appr = 1) M becomes small

        Args:
            rec (Record): Record instance containing every bit of information about the image
            k_blobs (int): number of biggest blobs to keep for the irregularity calculation
            
        Returns:
            float: mean M in [0, ∞), or np.nan if no valid blobs

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
    M_list = []

    # Iterate through each blob one-by-one
    for prop in props:
        # Isolating the current blob
        single = (labeled == prop.label)
        area = prop.area # Save the current blob's area
        perimeter = prop.perimeter # Save the current blob's perimeter
        
        # If the area or perimeter is zero, skip this blob
        if area <= 0 or perimeter <= 0:
            continue

        # Calculate the irregularity measure
        irregularity = (perimeter ** 2) / (4 * np.pi * area)

        # For the convexity measure, we need to assemble the coordinates of each pixel in the blob
        # np.nonzero(single) selects the True pixels in the current blob (it is needed, because the blob can be a eg.: a donut)
        # np.column_stack combines the x and y coordinates into a 2D array
        coords = np.column_stack(np.nonzero(single))
        
        # The convexhull function needs at least 3 seperate points to calculate the hull
        # If this condition is not met, we skip this blob
        if coords.shape[0] < 3:
            continue
        
        # ConvexHull finds the smallest convex polygon that encloses all of the points
        # Identifies the subset of points that lie on the convex hull, sorts them in counter-clockwise order, then defines line segments between successive hull vertices
        # Finally returns a polygon and we are using ".volume" to access its area
        convex_hull = ConvexHull(coords).volume
        
        # Fall back, to avoid division by zero in any case
        if convex_hull <= 0:
            continue
        
        # Using the formula to calculate the convexity measure
        convexity = area / convex_hull

        # Calculate the combined index M = I * (1 - C)
        M_list.append(irregularity * (1 - convexity))

    # If we have at least one valid blob, return the mean of M_list, otherwise NaN
    return float(np.nanmean(M_list)) if M_list else np.nan