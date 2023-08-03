import cv2
import numpy as np
import pytesseract
import re

room_num =[]
shape_type=[]
recognized_text =[]
sequence=[]
result=[]

def detect_black_circle(image_path):
    # Step 2: Read the image
    image = cv2.imread(image_path)
    height = image.shape[0]
    roi_start_y = 0
    roi_end_y = height // 3
    roi = image[roi_start_y:roi_end_y, :]
    # Step 3: Convert the image to grayscale
    gray_image = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    
    # Step 4: Apply Hough Circle Transform
    circles = cv2.HoughCircles(
        gray_image, 
        cv2.HOUGH_GRADIENT, 
        dp=1, 
        minDist=20, 
        param1=50, 
        param2=30, 
        minRadius=5, 
        maxRadius=100
    )
    black_circle_dict =[]
    white_circle_dict =[]
    bit_list=[]
    if circles is not None:
        # Step 5: Filter the detected circles
        circles = np.round(circles[0, :]).astype("int")
        i=0
        for (x, y, r) in circles:
            # Check if the circle is black
            circle_region = gray_image[y - r:y + r, x - r:x + r]
            mean_color = np.mean(circle_region)
            if mean_color < 150:  # Adjust the threshold to detect black circles accurately
                cv2.circle(image, (x, y), r, (0, 255, 0), 4)  # Draw green circle around the detected circle
                cv2.putText(image, f"({x}, {y})", (x, y -40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255,0), 2)
                cv2.putText(image, str(i+1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                i=i+1
                black_circle_dict.append(x)
            else:
                cv2.circle(image, (x, y), r, (255, 0, 0), 4)
                cv2.putText(image, f"({x}, {y})", (x, y+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(image, str(i+1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                i=i+1
                white_circle_dict.append(x)
                
    
    cv2.imshow("Detected Circles", image)


    all_value = black_circle_dict.copy()
    all_value.extend(white_circle_dict)
    sort_value = sorted(all_value)

    for value in sort_value:
        if value in black_circle_dict:
            bit_list.append(0)
        else:
            bit_list.append(1)
    #print(bit_list)
    
    for i in range(0,15,3):
        room_value = 0
        for j in range(2,-1,-1):
            if bit_list[i+j] == 1:
                room_value = room_value + 2**(2-j)
        room_num.append(room_value)
    return room_num        

def classify_shape(num_vertices):
    if num_vertices == 3:
        return "Triangle"
    elif num_vertices == 4:
        return "Square"
    elif num_vertices == 5:
        return "Pentagon"
    else:
        return "Circle"

def detect_shape(image_path):
    # Read the input image
    image = cv2.imread(image_path)

    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for blue color in the HSV color space
    lower_blue = np.array([90, 120, 120])
    upper_blue = np.array([130, 255, 255])

    # Threshold the image to extract blue regions
    blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

    # Apply morphological operations (optional but can help clean up the mask)
    kernel = np.ones((5, 5), np.uint8)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)

    # Find contours in the blue mask
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shape_dict={}
    for contour in contours:
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        num_vertices = len(approx)

        # Calculate the area of the contour
        area = cv2.contourArea(contour)

        # Classify shapes based on number of vertices
        shape_name = classify_shape(num_vertices)

        # Draw the shape name and area on the image
        moment = cv2.moments(contour)
        cx = int(moment['m10'] / moment['m00'])
        cy = int(moment['m01'] / moment['m00'])
        if (area < 4000):
            cv2.putText(image, f"Cross ({area:.2f})", (cx, cy-60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(image, f"({cx}, {cy})", (cx, cy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            shape_dict["Cross"]=cx

        else:
            cv2.putText(image, f"{shape_name} ({area:.2f})", (cx, cy-60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(image, f"({cx}, {cy})", (cx, cy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            shape_dict[shape_name]= cx


        # Draw the contour on the image
        cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)

    # Display the output image
    cv2.imshow('Detected Shapes', image)
    sort_shape_dict = sorted(shape_dict.items(), key=lambda x:x[1])
    shape_type = list(dict(sort_shape_dict))
    return shape_type

def detect_number(image_path):

    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    image = cv2.imread(image_path)

    height = image.shape[0]
    roi_start_y = 2*height//3
    roi_end_y = height 
    roi = image[roi_start_y:roi_end_y, :]

    # Convert to grayscale and apply Gaussian blur
    gray_image = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    erd = cv2.erode(gray_image,None,iterations=2)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Perform OCR to extract text from the image
    extracted_text = pytesseract.image_to_string(erd, config='--psm 6 outputbase digits')

    # Remove non-numeric characters
    recognized_text = re.sub(r'\D', '', extracted_text)

    # Get bounding boxes of the detected regions
    h, w, _ = image.shape
    boxes = pytesseract.image_to_boxes(erd, config='--psm 6 outputbase digits')
    for b in boxes.splitlines():
        b = b.split()
        x, y, x2, y2 = int(b[1]), h - int(b[2]), int(b[3]), h - int(b[4])
        cv2.rectangle(image, (x, y), (x2, y2), (0, 255, 0), 2)
    cv2.imshow('Image with Detection Box', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return recognized_text

# Replace 'path/to/your/image.jpg' with the actual path to your image
room_num= detect_black_circle('result.jpg')
shape_type = detect_shape('result.jpg')
recognized_text = detect_number('result.jpg')

cv2.waitKey(0)
cv2.destroyAllWindows()

#print(room_num)
#print(shape_type)
#print([*recognized_text])

for order in [*recognized_text]:
    if order == '1':
        sequence.append("First")
    elif order == '2':
        sequence.append("Second")
    elif order == '3':
        sequence.append("Third")
    elif order == '4':
        sequence.append("Fourth")
    else :
        sequence.append("Fifth")

for i in range(0,5):
    result.append([sequence[i],room_num[i],shape_type[i]])
print("ลำดับ | เลขห้อง | กล่อง")        

print(*result, sep = '\n')