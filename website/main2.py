# fall detection using opencv
import cv2
import time
#from sms import notification

# 
fitToEllipse = False
# to take input from pre recorded video in mp4 format
cap = cv2.VideoCapture('C:/Users/Asus.DESKTOP-1EO1OQ5/Desktop/7th Sem project/website/ArenaA.mp4')
# to take live input from webCam
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# the output will be written to output.avi
out = cv2.VideoWriter(
    'outputforfall.avi',
    cv2.VideoWriter_fourcc(*'MJPG'),
    15.,
    (640,480))
# dealy of 2 ms
time.sleep(2)
# Background Subtaction based on motion history
fgbg = cv2.createBackgroundSubtractorMOG2()
# counter to confirm fall event
count=0
# notification count
nc=0
while(1):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # print(ret)
    # Convert each frame to gray scale and subtract the background
    try:
        # resizing for faster detection
        frame = cv2.resize(frame, (640, 480))
        # using a greyscale picture, also for faster detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # foreground extraction
        fgmask = fgbg.apply(gray)
        # to display fgmask 
        # cv2.imshow("foreGroundSegmentation",fgmask)
        # cv2.waitKey(1)
        # Find contours
        contours, _ = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # if counters exists
        if contours:        
            # List to hold all areas
            areas = []
            # countors gives all moving objs
            for contour in contours:
                # calculating contour area
                ar = cv2.contourArea(contour)
                # appending all areas to area list
                areas.append(ar)
            # finding max area and storing in max_area
            max_area = max(areas, default = 0)
            # storng max area index in max_area_index
            max_area_index = areas.index(max_area)
            # storing max area contour in cnt
            cnt = contours[max_area_index]
            # 
            M = cv2.moments(cnt)
            # boundingReact function returns cordinates of four corners of contours
            x, y, w, h = cv2.boundingRect(cnt)
            # drawing Countoures
            cv2.drawContours(fgmask, [cnt], 0, (255,255,255), 3, maxLevel = 0)
            # if height is less then width
            if h < w:
                # count keeps the count
                count=count+1 
            # if count is gerater then 10     
            if count > 22:
                # print("FALL")
                # cv2.putText(fgmask, 'FALL', (x, y), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255,255,255), 2)
                # this function sends the SMS notification
                if(nc==0):
                    
                    # notification()
                    nc=nc+1
                cv2.putText(frame, 'Fall Detected',(x+100, y-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
            
            # if height is geater then width
            if h > w:
                count = 0
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            # Write the output video 
            out.write(frame.astype('uint8'))
            # displays video
            cv2.imshow('video', frame)
            key = cv2.waitKey(1)
            # to break loop when q is entered on keyboard
            if key == ord("q"):
              break

            # to break loop when there are no frames
            # if cv2.waitKey(33) == 27:
            # break
    # if there is any exception then print the exception and break       
    except Exception as e:
        print(e)
        break
# to close window after execution
cv2.destroyAllWindows()