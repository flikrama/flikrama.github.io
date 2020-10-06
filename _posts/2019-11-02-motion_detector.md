---
title: A motion detector using python3 and cv2 library
categories:
- cv2
- Computer Vision
- Python3


feature_image: /assets/img/motion_detector/owl.jpg
---
<iframe width = 900 height = 400 src="/Graph.html" title="Motion Detector"></iframe> 

Did you know you can build a motion detector using Python3 and cv2 library? Calling it a motion detector may be a bit of a stretch since it's actually just a change detector from the 1st video frame. Here's what the below code does:

1. It detects motion/changes from the 1st frame and highlights the area of the frame where change has occurred (with a green box)

2. It outputs an interactive/hoverable graph telling you when motion started and when ended (as many times as it occurs) which can serve as a summary of changes happening without having to review all the video frames.

<center><img src="/assets/img/motion_detector/frame.png"></center>
<center><i>Highlighting areas where there is motion</i></center>



**Workflow:**
1. Take the 1st video frame and save it

2. Extract the 1st frame from all the subsequent video frames an save it as a delta frame

3. Process this delta frame and dilate it

4. Based on a threshold (of size and magnitude) of difference decide whether it big enough to be considered a change or not

5. Draw a green rectangle on the area of change and save the times when change starts and when it ends into dataframe

6. Use the data from the dataframe and Bokeh library to plot



Here's the code below:

**The plotter function:**

```python

#Import libraries
import pandas
from bokeh.plotting import figure, show, output_file
from bokeh.models import HoverTool, ColumnDataSource

# Import the capture function defined below:
from capture import df


df['Start_string'] = df['Start'].dt.strftime('%Y-%m-%d %h %m %s')
df['End_string'] = df['End'].dt.strftime('%Y-%m-%d %h %m %s')
cds = ColumnDataSource(df)

p = figure(x_axis_type = 'datetime', height = 300, width  = 1500, 
title = 'Motion Sensor Graph')

#Define hovering tools
hover = HoverTool(tooltips = [("Start" , "@Start_string"),("End", "@End_string")])
p.add_tools(hover)
q = p.quad(left = 'Start', right = 'End', top = 1, 
bottom = 0, color = 'green', source = cds)

#Save html graph and display it
output_file('Graph.html')
show(p)
```

**The capture function where most of the work is done:**
``` python
#Import libraries
import cv2
import time
from datetime import datetime
import pandas

first_frame = None
status_list=[None, None]
times = []
df = pandas.DataFrame(columns = ['Start', 'End'])

video = cv2.VideoCapture(0)

while True:
    check, frame = video.read()
    status = 0
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21,21),0)
     
    if first_frame is None:
        first_frame = gray
        continue
    #Delta (change) frame from the 1st frame
    delta_frame = cv2.absdiff(first_frame, gray)
    # Apply a threshold such that it does not capture all the tiny changes
    thresh_frame = cv2.threshold(delta_frame, 75, 255, cv2.THRESH_BINARY)[1]
    # Dilate
    thresh_frame = cv2.dilate(thresh_frame, None, iterations = 3)
    # Find contours
    (cnts,_) = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    #Draw the rectangles:
    for contour in cnts:
        if cv2.contourArea(contour) < 15000:
            continue
        status = 1
        (x,y,w,h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x,y), (x+w, y+h),(0,255,0), 2)
    status_list.append(status)
    status_list = status_list[-2:] #clear all but last 2 items in list
    
    # Save times when an object comes in and goes out of the frame
    if status_list[-1] ==1 and status_list[-2] == 0:
        times.append(datetime.now())
    if status_list[-1] ==0 and status_list[-2] == 1:
        times.append(datetime.now())
    
    # Show the three frames defined above
    cv2.imshow('Gray Frame', gray)
    cv2.imshow('Delta Frame',delta_frame)
    cv2.imshow('Thresholded',thresh_frame)
    cv2.imshow('Color Frame', frame)
    key = cv2.waitKey(1)
    
    if key==ord('q'):
        if status ==1:
            times.append(datetime.now())
        break
# Put the times saved into a dataframe to be used by the plot function
for i in range(0, len(times),2):
    df = df.append({'Start':times[i], 'End': times[i+1]}, ignore_index = True)

# Save to a csv as well for records
df.to_csv('Times.csv')

#Destroy windows when prompted
video.release()
cv2.destroyAllWindows
```



<iframe width = 900 height = 400 src="/Graph.html" title="Motion Detector"></iframe>  


{% include button.html text="Github" icon="github" link="https://github.com/flikrama" color="#0366d6" %} {% include button.html text="Linkedin" icon="linkedin" link="https://www.linkedin.com/in/likrama/" color="#0e76a8" %}   [**Resume**](/assets/resume/Fatmir_Likrama.pdf)