"""
====================================================================
 RESTRICTED ZONE ANIMAL DETECTOR
====================================================================
This script watches a video for cats, dogs, and birds. If one of
these animals walks into a "restricted zone" that you define, the
script will draw a RED box around it and flash a warning message
on the screen. If the animal is anywhere else in the frame, it just
gets a normal GREEN box.

Libraries used:
  - ultralytics : gives us the YOLOv8 object detection model
  - opencv-python (cv2) : lets us read/write video files and draw
    shapes/text on video frames

Before running:
  pip install ultralytics opencv-python

Make sure a video file named "input_video.mp4" is in the same
folder as this script. The result will be saved as
"output_video.mp4" in that same folder.
====================================================================
"""

# ---- Step 1: Import the libraries we need -------------------------
import cv2                     # OpenCV: for reading/writing video and drawing shapes/text
from ultralytics import YOLO   # Ultralytics: gives us the YOLOv8 model class


# ====================================================================
# STEP 2: USER-EDITABLE SETTINGS
# ====================================================================
# These are the values a beginner is most likely to want to change.

# --- File names ---
INPUT_VIDEO_PATH = "input_video.mp4"     # the video we are analyzing
OUTPUT_VIDEO_PATH = "output_video.mp4"   # the video we will create

# --- Which YOLOv8 model to use ---
# "yolov8n.pt" is the "Nano" version: it's the smallest and fastest
# YOLOv8 model, which makes it great for quick tests and for running
# on machines without a powerful GPU. The ".pt" file will be
# automatically downloaded by ultralytics the first time you run this.
MODEL_NAME = "yolov8n.pt"

# --- Which animals (COCO dataset class IDs) do we care about? ---
# The YOLOv8 model was trained on the COCO dataset, where every
# object category has a fixed numeric ID. The ones we want here are:
#   14 = bird
#   15 = cat
#   16 = dog
# We store them in a Python set (a collection of unique values)
# because checking "is this ID in my set of interesting IDs?" is
# very fast and easy to read.
TARGET_CLASS_IDS = {14, 15, 16}

# A small dictionary just so we can print/display friendly animal
# names instead of raw numbers when we draw labels on the video.
CLASS_ID_TO_NAME = {
    14: "bird",
    15: "cat",
    16: "dog",
}

# --- Confidence threshold ---
# YOLO gives every detection a "confidence score" from 0.0 to 1.0,
# which tells us how sure the model is that it found a real object.
# We ignore any detection below this threshold to reduce false alarms.
CONFIDENCE_THRESHOLD = 0.4

# --- The Restricted Zone (a simple rectangle) ---
# This rectangle is defined by two corner points:
#   (ZONE_X_MIN, ZONE_Y_MIN)  -> the TOP-LEFT corner of the rectangle
#   (ZONE_X_MAX, ZONE_Y_MAX)  -> the BOTTOM-RIGHT corner of the rectangle
#
# Remember: in image/video coordinates, (0, 0) is the TOP-LEFT corner
# of the frame, x increases as you move RIGHT, and y increases as you
# move DOWN.
#
# >>> CHANGE THESE FOUR NUMBERS TO MOVE/RESIZE THE RESTRICTED ZONE <<<
# Example below places a rectangle roughly in the center-right of a
# 1280x720 video. Adjust the numbers to match your own video's
# resolution and the area you actually want to protect.
ZONE_X_MIN = 640   # left edge of the restricted zone (pixels from left)
ZONE_Y_MIN = 200   # top edge of the restricted zone (pixels from top)
ZONE_X_MAX = 1100  # right edge of the restricted zone (pixels from left)
ZONE_Y_MAX = 600   # bottom edge of the restricted zone (pixels from top)

# --- Colors (OpenCV uses BGR order, not RGB!) ---
COLOR_GREEN = (0, 255, 0)        # normal animal, outside the zone
COLOR_RED = (0, 0, 255)          # intruder animal, inside the zone
COLOR_BLUE = (255, 0, 0)         # used to highlight the restricted zone
COLOR_WHITE = (255, 255, 255)    # used for warning text

# How "see-through" the blue restricted-zone highlight should be.
# 0.0 = fully transparent (invisible), 1.0 = fully solid.
ZONE_OVERLAY_TRANSPARENCY = 0.3


# ====================================================================
# STEP 3: LOAD THE YOLOv8 MODEL
# ====================================================================
# This line creates a YOLO object and loads the pretrained weights
# from the "yolov8n.pt" file. If the file isn't already on disk,
# ultralytics will automatically download it for us.
print("Loading YOLOv8 model... this may take a moment the first time.")
model = YOLO(MODEL_NAME)


# ====================================================================
# STEP 4: OPEN THE INPUT VIDEO
# ====================================================================
# cv2.VideoCapture opens a video file so we can read it frame by frame,
# similar to how you'd open a text file to read it line by line.
video_capture = cv2.VideoCapture(INPUT_VIDEO_PATH)

# Always check that the video actually opened. If the file path is
# wrong or the file is corrupted, isOpened() will return False.
if not video_capture.isOpened():
    raise IOError(f"Could not open input video: {INPUT_VIDEO_PATH}")

# --- Read the input video's properties ---
# We need these so our OUTPUT video matches the INPUT video exactly
# (same width, same height, same frames-per-second).
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))    # video width in pixels
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))  # video height in pixels
frames_per_second = video_capture.get(cv2.CAP_PROP_FPS)           # how many frames play per second
total_frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))  # total frames in the video (for progress info)

print(f"Input video info -> width: {frame_width}, height: {frame_height}, "
      f"fps: {frames_per_second:.2f}, total frames: {total_frame_count}")


# ====================================================================
# STEP 5: PREPARE THE OUTPUT VIDEO WRITER
# ====================================================================
# cv2.VideoWriter is the object that lets us save frames as a new
# video file. We must tell it:
#   1) the output file name
#   2) the "codec" (compression format) to use
#   3) the frames-per-second for playback
#   4) the frame size (width, height) -- must match every frame we write

# "mp4v" is a widely-compatible codec for creating .mp4 files.
# fourcc stands for "Four Character Code" -- it's just a 4-letter
# code that identifies which video codec/compression method to use.
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

# Now we actually create the VideoWriter object using the same
# width, height, and FPS as the input video, so the output video
# matches it exactly.
video_writer = cv2.VideoWriter(
    OUTPUT_VIDEO_PATH,
    fourcc,
    frames_per_second,
    (frame_width, frame_height)
)


# ====================================================================
# STEP 6: HELPER FUNCTION - is a point inside the restricted zone?
# ====================================================================
def is_point_inside_zone(point_x, point_y):
    """
    Returns True if the given (point_x, point_y) coordinate falls
    inside the restricted rectangle we defined above, otherwise False.
    """
    inside_x_range = ZONE_X_MIN <= point_x <= ZONE_X_MAX
    inside_y_range = ZONE_Y_MIN <= point_y <= ZONE_Y_MAX
    return inside_x_range and inside_y_range


# ====================================================================
# STEP 7: MAIN LOOP - process the video one frame at a time
# ====================================================================
# We keep a running counter of which frame number we're on, purely
# so we can print progress messages to the terminal.
current_frame_number = 0

print("Starting video processing...")

# This while loop keeps running until we run out of frames to read.
while True:

    # video_capture.read() grabs the NEXT frame from the video.
    # It returns two things:
    #   frame_was_read : True if a frame was successfully grabbed,
    #                     False if we've reached the end of the video
    #   current_frame  : the actual image data (a NumPy array) for
    #                     that frame, which we can draw on
    frame_was_read, current_frame = video_capture.read()

    # If there are no more frames left, break out of the loop.
    if not frame_was_read:
        break

    current_frame_number += 1

    # ----------------------------------------------------------------
    # STEP 7a: Draw the semi-transparent blue restricted zone
    # ----------------------------------------------------------------
    # To make a shape "semi-transparent", the common OpenCV trick is:
    #   1) Make a full copy of the current frame ("overlay")
    #   2) Draw a SOLID rectangle on that copy
    #   3) Blend the copy back into the original frame using a
    #      transparency ("alpha") value with cv2.addWeighted

    overlay = current_frame.copy()  # a duplicate of the frame we can draw solid shapes on

    # cv2.rectangle draws a rectangle on an image.
    #   overlay                -> the image to draw on
    #   (ZONE_X_MIN, ZONE_Y_MIN) -> top-left corner point
    #   (ZONE_X_MAX, ZONE_Y_MAX) -> bottom-right corner point
    #   COLOR_BLUE              -> the color of the rectangle (BGR)
    #   -1                      -> thickness of -1 means "fill the
    #                              rectangle solid" instead of just
    #                              drawing its outline
    cv2.rectangle(
        overlay,
        (ZONE_X_MIN, ZONE_Y_MIN),
        (ZONE_X_MAX, ZONE_Y_MAX),
        COLOR_BLUE,
        -1
    )

    # cv2.addWeighted blends two images together using weights.
    # Here we blend the solid-blue "overlay" with the original
    # "current_frame" so that the blue rectangle appears see-through.
    #   ZONE_OVERLAY_TRANSPARENCY       -> weight of the overlay image
    #   1 - ZONE_OVERLAY_TRANSPARENCY   -> weight of the original frame
    #   0                                -> a brightness offset (unused, so 0)
    # The blended result is written back into current_frame.
    cv2.addWeighted(
        overlay,
        ZONE_OVERLAY_TRANSPARENCY,
        current_frame,
        1 - ZONE_OVERLAY_TRANSPARENCY,
        0,
        current_frame
    )

    # ----------------------------------------------------------------
    # STEP 7b: Run YOLOv8 detection on this single frame
    # ----------------------------------------------------------------
    # model(...) runs the neural network on the frame and returns a
    # list of "Results" objects (one per image we passed in -- since
    # we only passed one frame, we only get one Results object back).
    #   conf=CONFIDENCE_THRESHOLD -> ignore low-confidence detections
    #   verbose=False              -> stop YOLO from printing a log
    #                                 line for every single frame
    detection_results = model(current_frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

    # We only passed in one frame, so we grab the first (and only)
    # result from the list.
    result = detection_results[0]

    # Flag to track whether ANY intruder animal was found in this
    # frame, so we know whether to draw the big warning text.
    intruder_found_in_this_frame = False

    # ----------------------------------------------------------------
    # STEP 7c: Loop through every object YOLO detected in this frame
    # ----------------------------------------------------------------
    # result.boxes contains one entry per detected object. Each
    # "box" holds the bounding box coordinates, confidence score,
    # and predicted class ID.
    for box in result.boxes:

        # box.cls holds the predicted class ID (e.g. 15 for "cat").
        # It comes back as a small tensor, so we convert it to a
        # plain Python integer with int(...).
        detected_class_id = int(box.cls[0])

        # Skip this detection entirely if it's not one of our target
        # animals (cat, dog, bird). This is our "filtering" step.
        if detected_class_id not in TARGET_CLASS_IDS:
            continue

        # box.xyxy gives us the bounding box as
        # [x_min, y_min, x_max, y_max] in pixel coordinates.
        # We convert it to plain integers for drawing.
        x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

        # Calculate the CENTER point of the bounding box. We use the
        # center (not the corners) to decide whether the animal is
        # "inside" the restricted zone, since the center is a more
        # reliable stand-in for "where the animal's body actually is".
        box_center_x = (x_min + x_max) // 2
        box_center_y = (y_min + y_max) // 2

        # Look up a friendly name like "cat" for the label text.
        animal_name = CLASS_ID_TO_NAME.get(detected_class_id, "animal")

        # Grab the confidence score (how sure YOLO is) so we can
        # show it in the label, formatted as a percentage.
        confidence_score = float(box.conf[0])

        # ------------------------------------------------------------
        # STEP 7d: Decide if this animal is inside the restricted zone
        # ------------------------------------------------------------
        if is_point_inside_zone(box_center_x, box_center_y):
            # The animal's center is inside the restricted rectangle,
            # so this is an "intruder" -> draw its box in RED.
            box_color = COLOR_RED
            intruder_found_in_this_frame = True
        else:
            # The animal is outside the restricted zone -> draw GREEN.
            box_color = COLOR_GREEN

        # --- Draw the bounding box around the animal ---
        # cv2.rectangle here uses thickness=2 (a normal outlined box,
        # not filled solid like the zone overlay was).
        cv2.rectangle(current_frame, (x_min, y_min), (x_max, y_max), box_color, 2)

        # --- Draw a text label just above the bounding box ---
        label_text = f"{animal_name} {confidence_score:.0%}"
        cv2.putText(
            current_frame,
            label_text,
            (x_min, max(y_min - 10, 0)),  # position slightly above the box; avoid negative coords
            cv2.FONT_HERSHEY_SIMPLEX,     # a standard, readable font
            0.6,                          # font scale (size)
            box_color,                    # text color matches the box color
            2                             # text thickness
        )

    # ----------------------------------------------------------------
    # STEP 7e: If any intruder was found, draw a big warning banner
    # ----------------------------------------------------------------
    if intruder_found_in_this_frame:
        warning_message = "WARNING: ANIMAL IN RESTRICTED AREA"

        # Draw a solid black rectangle behind the text first, so the
        # white warning text is always easy to read no matter what's
        # in the background of the video.
        cv2.rectangle(current_frame, (0, 0), (frame_width, 40), (0, 0, 0), -1)

        cv2.putText(
            current_frame,
            warning_message,
            (10, 28),                     # position near the top-left of the frame
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,                           # slightly bigger font for visibility
            COLOR_WHITE,
            2
        )

    # ----------------------------------------------------------------
    # STEP 7f: Write this finished frame to the output video file
    # ----------------------------------------------------------------
    video_writer.write(current_frame)

    # Print progress every 30 frames so the user knows it's working.
    if current_frame_number % 30 == 0:
        print(f"Processed frame {current_frame_number}/{total_frame_count}")


# ====================================================================
# STEP 8: CLEAN UP - release the video files
# ====================================================================
# It's important to "release" both the reader and writer when we're
# done. This closes the files properly and finalizes/saves the
# output video to disk. Forgetting this step can result in a
# corrupted or unplayable output video.
video_capture.release()
video_writer.release()

print(f"Done! Processed {current_frame_number} frames.")
print(f"Output video saved to: {OUTPUT_VIDEO_PATH}")