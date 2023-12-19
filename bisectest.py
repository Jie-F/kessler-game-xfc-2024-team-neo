import bisect

def find_extreme_shooting_angle_error(asteroid_list, threshold, mode='largest_below'):
    # Extract the shooting_angle_error_deg values
    shooting_angles = [d['shooting_angle_error_deg'] for d in asteroid_list]

    if mode == 'largest_below':
        # Find the index where threshold would be inserted
        idx = bisect.bisect_left(shooting_angles, threshold)

        # Adjust the index to get the largest value below the threshold
        if idx > 0:
            idx -= 1
        else:
            return None  # All values are greater than or equal to the threshold
    elif mode == 'smallest_above':
        # Find the index where threshold would be inserted
        idx = bisect.bisect_right(shooting_angles, threshold)

        # Check if all values are smaller than the threshold
        if idx >= len(shooting_angles):
            return None
    else:
        raise ValueError("Invalid mode. Choose 'largest_below' or 'smallest_above'.")

    # Return the corresponding dictionary
    return asteroid_list[idx]

# Example usage:
list_of_dicts = [
    {"shooting_angle_error_deg": -120},
    {"shooting_angle_error_deg": -60},
    {"shooting_angle_error_deg": -45},
    {"shooting_angle_error_deg": -39.999999},
    {"shooting_angle_error_deg": -30},
    {"shooting_angle_error_deg": -10},
    {"shooting_angle_error_deg": 25},
    {"shooting_angle_error_deg": 45},
    {"shooting_angle_error_deg": 49.999},
    {"shooting_angle_error_deg": 52},
    {"shooting_angle_error_deg": 59},
    {"shooting_angle_error_deg": 80},
]

# For finding the largest value below a positive threshold
turn_angle_deg_until_can_fire = 50
result = find_extreme_shooting_angle_error(list_of_dicts, turn_angle_deg_until_can_fire, mode='largest_below')
print("Largest below threshold:", result)

# For finding the smallest value above a negative threshold
negative_threshold = -40
result = find_extreme_shooting_angle_error(list_of_dicts, negative_threshold, mode='smallest_above')
print("Smallest above threshold:", result)
