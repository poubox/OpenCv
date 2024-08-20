
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2


class LaneDetect():
    def __init__(self):
        self.right_fit = []
        self.left_fit = []
            
    def CvImage(self,image):
        

        
        try:
            # img = cv2.imread(frame, cv2.IMREAD_ANYCOLOR)
            # img = cv2.resize(img,(400,256))
            img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        except :
            print("Oops, image isn't working!")
            
        def sobel(img, dx, dy):
            img_sobel = cv2.Sobel(img, -1, dx, dy, delta=0)
            return img_sobel
        dx = 0
        dy = 1
        img_sobel = sobel(img_hsv, dx, dy)
        
        # 영상에서 특정 영역의 색상만 추출시킨다.
        def bird(img):
            imshape = img.shape

            width = imshape[1]
            height = imshape[0]

            pts1 = np.float32([[width*4/10, height*6/10],[width*3/20, height - 100],[width*6.5/10,height*6/10],[width*9/10, height - 100]]) 

            # pts2의 좌표 지정
            pts2 = np.float32([[0,0],[0,height],[width*9/10,0],[width*9/10,height]])

            M = cv2.getPerspectiveTransform(pts1, pts2)

            img = cv2.warpPerspective(img, M, (width,1000))

            h, s, v = cv2.split(img)

            new_img = cv2.merge((h, s, v))

            return new_img

        new_img = bird(img_sobel)

 
        def inRange(img):
            low_white = np.array([0, 0, 180], dtype="uint8")
            high_white = np.array([30, 30, 255], dtype="uint8")
            low_yellow = np.array([15, 150, 150], dtype="uint8")
            high_yellow = np.array([30, 255, 200], dtype="uint8")

            dst_white = cv2.inRange(img, low_white, high_white)
            dst_yellow = cv2.inRange(img, low_yellow, high_yellow)
            dst = dst_white + dst_yellow
            return dst
    
        edges = inRange(new_img)
        
        def canny(img):
            img = cv2.Canny(img, 5000, 1500, apertureSize=5, L2gradient= True)
            return img
            
        edges = canny(edges)
        # cv2.imshow("de", edges)
        # cv2.waitKey()

        def region_of_interest(img, vertices):

            mask = np.zeros_like(img)

            if len(img.shape) > 2:
                channel_count = img.shape[2]
                ignore_mask_color = (255,) * channel_count
            else:
                ignore_mask_color = 255

            cv2.fillPoly(mask, vertices, ignore_mask_color)

            masked_image = cv2.bitwise_and(img, mask)
            
            return masked_image

        new_imshape = edges.shape

        height = new_imshape[0]
        width = new_imshape[1]

        vertices = np.array([[(0, height),
                                (width/10,height/3),
                                (width,height/3),
                                (width, height)]], dtype=np.int32)

        mask = region_of_interest(edges, vertices)
        def findcontours(img):
            contours, h = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            return contours
        
        contours = findcontours(mask)
        
        def draw_contours(img, contours):
            color = [255, 255, 255]
            for i, contour in enumerate(contours):
                cv2.drawContours(img, [contour], -1, (100, 30, 10), thickness=3)
            return img
                
        mask = draw_contours(mask, contours)
        
        
        
        def draw_lines(img, lines, color=[125, 125, 255], thickness=5):
            avg_x = []
            avg_y = []

            if lines is not None:

                for line in lines:
                    for x1, y1, x2, y2 in line:
                        ratio = (y2 - y1)/(x2 - x1)
                        # ratio = np.clip(ratio, -1, 1)
                        if (x2 - x1) == 0:
                            pass
                        elif ratio > 1 or ratio < -1:
                            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
                            avg_x_val = (x1+x2)/2
                            avg_y_val = (y1+y2)/2
                            avg_x.append(avg_x_val)
                            avg_y.append(avg_y_val)
                            
                        elif np.arccos(ratio) < 10: 
                            pass
                        
                        else:
                            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
                            avg_x_val = (x1+x2)/2
                            avg_y_val = (y1+y2)/2
                            avg_x.append(avg_x_val)
                            avg_y.append(avg_y_val)

                # cv2.line(img, (avg_x[1], avg_y[1]), (avg_x[-1], avg_y[-1]), color, thickness)


        def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
            lines = cv2.HoughLinesP(img, rho, theta, threshold, minLineLength=min_line_len, maxLineGap=max_line_gap)
            line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
            draw_lines(line_img, lines)

            return line_img

        rho = 1.0
        theta = np.pi/180
        threshold = 100
        min_line_len = 30
        max_line_gap = 80
        

        lines = hough_lines(mask, rho, theta, threshold, min_line_len, max_line_gap)
        
        def window_sliding(img):
            center_x = [] # 중앙값 초기화
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 이미지를 grayscale로 변환
            hist = np.sum(img[:, :], axis = 0) # 이미지의 픽셀값을 히스토그램으로 변환
            
            out = np.dstack((img, img, img)) * 255 
            
            mid = int(hist.shape[0] / 2) # 히스토그램의 중심점 설정
            left_x = np.argmax(hist[:mid]) # 왼쪽 부분에서 히스토그램이 최댓값이 되는 x값을 찾고
            right_x = np.argmax(hist[mid:]) + mid # 오른쪽 부분에서 히스토그램이 최댓값이 되는 x값을 찾고

            
            num_window = 20 # 윈도우의 갯수 설정
            window_h = int(img.shape[0] / num_window) # 윈도우의 높이는 이미지의 높이/윈도우의 갯수
            
            nonzero = img.nonzero() # 이미지에서 픽셀값이 0이 아닌 좌표를 기록
            nonzero_y = np.array(nonzero[0]) # 그 좌표중에 y값
            nonzero_x = np.array(nonzero[1]) # 그 좌표중에 x값

            now_left_x = left_x
            now_right_x = right_x

            min_pixel = 60
            window_width = 20
            
            win_left_lane = []
            win_right_lane = []


            for window in range(num_window):
                win_y_low = img.shape[0] - (window + 1) * window_h
                win_y_high = img.shape[0] - window * window_h
                win_left_xmin = now_left_x - window_width
                win_left_xmax = now_left_x + window_width
                win_right_xmin = now_right_x - window_width
                win_right_xmax = now_right_x + window_width   
                
                
                cv2.rectangle(out, (win_left_xmin, win_y_low), (win_left_xmax, win_y_high), (0, 255, 0), 2)
                cv2.rectangle(out, (win_right_xmin, win_y_low), (win_right_xmax, win_y_high), (0, 255, 0), 2)  
                left_window_idx = ((nonzero_y >= win_y_low) & (nonzero_y <= win_y_high) & (nonzero_x >= win_left_xmin) & (
                    nonzero_x <= win_left_xmax)).nonzero()[0]
                right_window_idx = ((nonzero_y >= win_y_low) & (nonzero_y <= win_y_high) & (nonzero_x >= win_right_xmin) & (
                    nonzero_x <= win_right_xmax)).nonzero()[0]
                # Append these indices to the lists
                win_left_lane.append(left_window_idx)
                win_right_lane.append(right_window_idx)

                # If you found > minpix pixels, recenter next window on their mean position
                if len(left_window_idx) > min_pixel:
                    now_left_x = int(np.mean(nonzero_x[left_window_idx]))
                if len(right_window_idx) > min_pixel:
                    now_right_x = int(np.mean(nonzero_x[right_window_idx]))   
                
            win_left_lane = np.concatenate(win_left_lane)
            win_right_lane = np.concatenate(win_right_lane)

            # Extract left and right line pixel positions
            left_x, left_y = nonzero_x[win_left_lane], nonzero_y[win_left_lane]
            right_x, right_y = nonzero_x[win_right_lane], nonzero_y[win_right_lane]

            
            if len(left_x) > len(right_x):
                center_x = [(x + y)/2 for x, y in zip(left_x[:len(right_x)-1], right_x)]

            elif len(left_x) < len(right_x):
                center_x = [(x + y)/2 for x, y in zip(right_x[:len(right_x)-1], left_x)]


            # try:
            #     cv2.line(out, (int(center_x[0]), 100), (int(center_x[-1]), 900), [255, 255, 255], thickness=10)
            # except:
            #     pass
            return out, left_x, right_x, left_y, right_y, center_x

        out, left_x, right_x, left_y, right_y, center_x = window_sliding(lines)
        
        def polyfit(left_x, right_x, left_y, right_y):

            if len(left_x) != 0 and len(right_x) != 0 and len(left_y) != 0 and len(right_y) != 0:
                left_fit = np.polyfit(left_y, left_x, 3)
                right_fit = np.polyfit(right_y, right_x, 3)
            else:

                if len(left_x) != 0 or len(left_y) != 0:
                     right_fit = []
                     left_fit = np.polyfit(left_y, left_x, 3)
                elif len(right_fit) != 0 or len(right_y) != 0:
                    left_fit = []
                    right_fit = np.polyfit(right_y, right_x, 3)
                else:
                    left_fit = []
                    right_fit = []
                    pass

            left_fit = np.array(left_fit, dtype=np.float32)
            right_fit = np.array(right_fit, dtype=np.float32)
            
            return left_fit, right_fit
        
        
        def Output(self, out, left_x, right_x, left_y, right_y):
            right_fit = self.right_fit
            left_fit = self.left_fit
            left_detect = 0 # 왼쪽 차선이 detect 되면 1, 안되면 0
            right_detect = 0 # 오른쪽 차선이 detect 되면 1, 안되면 0
            
            if len(left_x) != 0  and len(right_x) != 0   and len(left_y) != 0   and len(right_y) != 0  :
                cv2.putText(out, "Left lane is detecting",(100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), thickness=2)
                cv2.putText(out, "Right lane is detecting",(800, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), thickness=2)
                left_fit, right_fit = polyfit(left_x, right_x, left_y, right_y)
                left_detect = 1 
                right_detect = 1 
                # print("left_fit = \n",left_fit)
                # print("right_fit = \n",right_fit)
                # print("left_fit[0] = \n",type(left_fit[0]))
                # print("\n")
            else:
                left_fit, right_fit = polyfit(left_x, right_x, left_y, right_y)
                if len(right_x) == 0 or len(right_y) == 0:

                    cv2.putText(out, "Left lane is detecting",(100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), thickness=2)
                    left_detect = 1 
                    right_detect = 0                
                elif len(left_x) == 0 or len(left_y) == 0:
                    cv2.putText(out, "Right lane is detecting",(800, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), thickness=2)
                    left_detect = 0 
                    right_detect = 1                
                else:
                    left_detect = 0 
                    right_detect = 0
                    pass
            
            return left_fit, right_fit, left_detect, right_detect
        
        left_fit, right_fit, left_detect, right_detect= Output(self, out, left_x, right_x, left_y, right_y)

        # cv2.imshow("original",out)
        return out, left_fit, right_fit, left_detect, right_detect


def main(args=None):
    
    lane_detect = LaneDetect()
    movie_link = "./movie/lane.mp4"
    
    
    cap = cv2.VideoCapture(movie_link)
    
    if cap.isOpened():
        
        while True:
            ret, frame = cap.read()
            out, left_fit, right_fit, left_detect, right_detect = lane_detect.CvImage(frame)
            
            
            cv2.imshow("lane detect",out)
            if not ret:
                print("No movie")
                break                
                
            if cv2.waitKey(40) == ord("q"):
                break    
    cap.release()
    cv2.destroyAllWindows()    


if __name__ == "__main__":
    main()
    


