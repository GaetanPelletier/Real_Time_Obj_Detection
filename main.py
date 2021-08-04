import cv2
import time
import torch
import functions_for_obj_detection as fct

#----------------------#

def main():

    # to get the frame rate
    pTime = 0   # previous time
    cTime = 0   # current time

    video_cap = cv2.VideoCapture(0) # number of the webcam, here the first

    # Model
    model_detection = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    # Using the cam
    while True:
        # get the frame
        success, img = video_cap.read()

        if success:
            new_img = fct.draw_anchor_box(img, fct.detection(img, model_detection))

            # get the frame rate
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime

            # display the fps on the screen
            cv2.putText(
                new_img,                    # draw on the screen
                str(int(fps)),          # round fps
                (10, 70),               # text position
                cv2.FONT_HERSHEY_PLAIN, # font type
                3,                      # fonct scale
                (255, 0, 255),          # color purple
                3                       # thickness
            )

            # show the frame
            cv2.imshow("Img captured", new_img)
            # wait before going to the next frame
            cv2.waitKey(1)
        
        else:
            print("No cam detected...")

#----------------------#

# if we are running this scipt,
if __name__ == "__main__":
    # then do this:
    main()