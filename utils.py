def normalize_landmarks(landmarks):
    """
    Normalizes landmarks relative to the wrist (landmark 0) and scales to [-1, 1].
    landmarks: list of [x, y, z]
    """
    if not landmarks:
        return []
    
    # Landmark 0 is the wrist
    wrist = landmarks[0]
    
    # Step 1: Subtract wrist coordinates from all coordinates
    rel_landmarks = []
    for lm in landmarks:
        rel_landmarks.append([
            lm[0] - wrist[0],
            lm[1] - wrist[1],
            lm[2] - wrist[2]
        ])
    
    # Step 2: Flatten the list
    flat_landmarks = []
    for p in rel_landmarks:
        flat_landmarks.extend(p)
        
    # Step 3: Scale by max absolute value to ensure range is roughly [-1, 1]
    max_val = max([abs(f) for f in flat_landmarks])
    if max_val != 0:
        flat_landmarks = [f / max_val for f in flat_landmarks]
        
    return flat_landmarks
