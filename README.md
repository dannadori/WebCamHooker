# WebCamHooker
- Web link(Japanese): https://cloud.flect.co.jp/entry/2020/03/31/162537
- Next link: see README2.md

if you want to know about conversion to anime, see README3.md

<p align="center">
<img src="./doc/fps_test2.gif" width="800" />
</p>




## 1. WebCamHooker

Webcamhooker takes the input of a physical webcam and outputs it to a virtual webcam, which can be used to modify images in video chats such as Teams and Zoom.

<p align="center">
<img src="./doc/out.gif" width="800" />
</p>

## 2. Prerequisite
It should work fine on most Linux systems, but the environment I worked in is a Debian Buster.
```
$ cat /etc/debian_version
10.3
```

Also, if you don't seem to have python3 on board, please introduce it.
```
$ python3 --version
Python 3.7.3
```

#### Install related software.
##### Virtual Webcam Device
This time, we will use a virtual webcam device called v4l2loopback.
https://github.com/umlaeute/v4l2loopback


We need to identify the virtual webcam device and the actual webcam, so we first check the device file of the actual webcam.
In the example below, it looks like video0 and video1 are assigned to the actual webcam.
```
$ ls /dev/video*.
/dev/video0 /dev/video1
```

So, let's introduce v4l2loopback.
First of all, please git clone, make and install.
```
$ git clone https://github.com/umlaeute/v4l2loopback.git
$ cd v4l2loopback
$ make
$ sudo make install
```
Next, load the module. In this case, it is necessary to add exclusive_caps=1 to make it recognized by chrome. [https://github.com/umlaeute/v4l2loopback/issues/78]
```
sudo modprobe v4l2loopback exclusive_caps=1
```
Now that the module is loaded, let's check the device file. In the example below, video2 has been added.
```
$ ls /dev/video*.
/dev/video0 /dev/video1 /dev/video2
```

##### ffmpeg
The easiest way to send data to a virtual webcam device is to use ffmpeg.
You can use apt-get and so on to introduce it quickly.

### Web camera hooks and video delivery
This time, I'm going to do some image processing once it detects a smile.
When a smile is detected, a smile symbol will be displayed on the video.

First, clone the following repository files to install the module.
```
$ git clone https://github.com/dannadori/WebCamHooker.git
$ cd WebCamHooker/
$ pip3 install -r requirements.txt
```

Here you will get the cascade file, you can find out more about cascade file in opencv official.
https://github.com/opencv/opencv/tree/master/data/haarcascades
```
$ wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml -P models
$ wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_smile.xml -P models
```

Let's borrow the smile mark from this site.
```
$ wget https://4.bp.blogspot.com/-QeM2lPMumuo/UNQrby-TEPI/AAAAAAAAI7E/cZIpq3TTyas/s160/mark_face_laugh.png -P images
```

I hope it has a folder structure like this.
```
$ ls -1
haarcascade_frontalface_default.xml
haarcascade_smile.xml
mark_face_laugh.png
webcamhooker.py
```

The execution is as follows.
 ** --input_video_num should be the actual webcam device number. For /dev/video0, enter a trailing 0.
 ** --output_video_dev must be the device file of the virtual webcam device.
In addition, please use ctrl+c to terminate.
```
$ python3 webcamhooker.py --input_video_num 0 --output_video_dev /dev/video2
```

When the above command is executed, ffmpeg starts to run and the video is delivered to the virtual camera device.

#### Let's have a video chat!
When you want to have a video chat, you should see something called "dummy~" in the list of video devices, so select it.
