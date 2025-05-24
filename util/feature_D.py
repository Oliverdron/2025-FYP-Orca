# Hair feature extraction code - Etele

#hair_pixel_count = np.count_nonzero(thresh)
#    total_pixels = thresh.shape[0] * thresh.shape[1]
#
#    # Hair proportion
#    hair_ratio = hair_pixel_count / total_pixels
#
#    # Normalize to 0–2 range
#    score = np.clip(hair_ratio * 6, 0, 2)
#    #score = np.clip(np.sum(blackhat / 255) / (total_pixels/30), 0, 2) --- weight based on blackhat pixel count
#    if score < 0.75:
#        label = 0
#    elif score < 1.5:
#        label = 1
#    else:
#        label = 2